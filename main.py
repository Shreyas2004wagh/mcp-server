from typing import Annotated
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
import markdownify
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, TextContent
from openai import BaseModel
from pydantic import AnyUrl, Field
import readabilipy
from pathlib import Path
import docx2txt
import PyPDF2
import io

# TODO: Replace these with your actual values
TOKEN = "XXXXXXXXXXX"
MY_NUMBER = "XXXXXXXX"  # Insert your number {91}{Your number}

class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None

class SimpleBearerAuthProvider(BearerAuthProvider):
    """
    A simple BearerAuthProvider that does not require any specific configuration.
    It allows any valid bearer token to access the MCP server.
    """

    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(
            public_key=k.public_key, jwks_uri=None, issuer=None, audience=None
        )
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="unknown",
                scopes=[],
                expires_at=None,
            )
        return None

class Fetch:
    IGNORE_ROBOTS_TXT = True
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        """
        Fetch the URL and return the content in a form ready for the LLM.
        """
        from httpx import AsyncClient, HTTPError

        async with AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except HTTPError as e:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"
                    )
                )
            if response.status_code >= 400:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR,
                        message=f"Failed to fetch {url} - status code {response.status_code}",
                    )
                )

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = (
            "<html" in page_raw[:100] or "text/html" in content_type or not content_type
        )

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(
            html, use_readability=True
        )
        if not ret["content"]:
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(
            ret["content"],
            heading_style=markdownify.ATX,
        )
        return content

# Initialize MCP server
mcp = FastMCP(
    "My MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

ResumeToolDescription = RichToolDescription(
    description="Serve your resume in plain markdown.",
    use_when="Puch (or anyone) asks for your resume; this must return raw markdown, no extra formatting.",
    side_effects=None,
)

def convert_file_to_markdown(file_path: str) -> str:
    """Convert various file formats to markdown text."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {file_path}")
    
    file_extension = path.suffix.lower()
    
    try:
        if file_extension == '.pdf':
            # Handle PDF files with multiple extraction methods
            text = ""
            
            # Try PyPDF2 first
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as pdf_error:
                print(f"PyPDF2 failed: {pdf_error}")
                
            # If PyPDF2 didn't extract text, try pdfplumber
            if not text.strip():
                try:
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                except ImportError:
                    print("pdfplumber not available, install with: pip install pdfplumber")
                except Exception as plumber_error:
                    print(f"pdfplumber failed: {plumber_error}")
            
            # If still no text, try pymupdf (fitz)
            if not text.strip():
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(file_path)
                    for page in doc:
                        page_text = page.get_text()
                        if page_text:
                            text += page_text + "\n"
                    doc.close()
                except ImportError:
                    print("PyMuPDF not available, install with: pip install pymupdf")
                except Exception as fitz_error:
                    print(f"PyMuPDF failed: {fitz_error}")
            
            if not text.strip():
                raise Exception("Could not extract text from PDF. The PDF might be image-based or corrupted.")
            
            return text.strip()
                
        elif file_extension in ['.docx', '.doc']:
            # Handle Word documents
            text = docx2txt.process(file_path)
            if not text.strip():
                raise Exception("Could not extract text from Word document")
            return text.strip()
            
        elif file_extension == '.txt':
            # Handle plain text files
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
                
        elif file_extension == '.md':
            # Handle markdown files
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
                
        else:
            # Try to read as plain text for other formats
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read().strip()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read().strip()
                
    except Exception as e:
        raise Exception(f"Error reading resume file: {str(e)}")

@mcp.tool(description=ResumeToolDescription.model_dump_json())
async def resume() -> str:
    """
    Return your resume exactly as markdown text.
    """
    try:
        # TODO: Update this path to point to your actual resume file
        # Common resume file locations to check:
        resume_paths = [
            "./resume.pdf",
            "./resume.docx",
            "./resume.txt",
            "./resume.md",
            "./CV.pdf",
            "./CV.docx",
            "~/Documents/resume.pdf",
            "~/Documents/resume.docx",
            "~/Desktop/resume.pdf",
            "~/Desktop/resume.docx"
        ]
        
        # Try to find the resume file
        resume_file = None
        for path in resume_paths:
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                resume_file = str(expanded_path)
                print(f"Found resume file: {resume_file}")
                break
        
        if not resume_file:
            print("No resume file found in common locations")
            # If no resume file found, return an error message
            return "ERROR: No resume file found. Please place your resume file (resume.pdf, resume.docx, resume.txt, or resume.md) in the same directory as the server script."
        
        # Convert the found file to markdown
        print(f"Attempting to convert {resume_file} to markdown...")
        markdown_content = convert_file_to_markdown(resume_file)
        
        # Validate that we got meaningful content
        if not markdown_content or len(markdown_content.strip()) < 50:
            return f"ERROR: Resume file found but content is too short or empty. File: {resume_file}, Content length: {len(markdown_content) if markdown_content else 0}"
        
        print(f"Successfully extracted {len(markdown_content)} characters from resume")
        return markdown_content
        
    except Exception as e:
        error_msg = f"Error loading resume: {str(e)}. Please ensure your resume file exists and is accessible."
        print(error_msg)
        return error_msg

@mcp.tool
async def validate() -> str:
    """
    NOTE: This tool must be present in an MCP server used by puch.
    """
    return MY_NUMBER

FetchToolDescription = RichToolDescription(
    description="Fetch a URL and return its content.",
    use_when="Use this tool when the user provides a URL and asks for its content, or when the user wants to fetch a webpage.",
    side_effects="The user will receive the content of the requested URL in a simplified format, or raw HTML if requested.",
)

@mcp.tool(description=FetchToolDescription.model_dump_json())
async def fetch(
    url: Annotated[AnyUrl, Field(description="URL to fetch")],
    max_length: Annotated[
        int,
        Field(
            default=5000,
            description="Maximum number of characters to return.",
            gt=0,
            lt=1000000,
        ),
    ] = 5000,
    start_index: Annotated[
        int,
        Field(
            default=0,
            description="On return output starting at this character index.",
            ge=0,
        ),
    ] = 0,
    raw: Annotated[
        bool,
        Field(
            default=False,
            description="Get the actual HTML content if the requested page, without simplification.",
        ),
    ] = False,
) -> list[TextContent]:
    """Fetch a URL and return its content."""
    url_str = str(url).strip()
    if not url:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="URL is required"))

    content, prefix = await Fetch.fetch_url(url_str, Fetch.USER_AGENT, force_raw=raw)
    original_length = len(content)
    if start_index >= original_length:
        content = "<error>No more content available.</error>"
    else:
        truncated_content = content[start_index : start_index + max_length]
        if not truncated_content:
            content = "<error>No more content available.</error>"
        else:
            content = truncated_content
            actual_content_length = len(truncated_content)
            remaining_content = original_length - (start_index + actual_content_length)
            if actual_content_length == max_length and remaining_content > 0:
                next_start = start_index + actual_content_length
                content += f"\n\n<error>Content truncated. Call the fetch tool with a start_index of {next_start} to get more content.</error>"
    return [TextContent(type="text", text=f"{prefix}Contents of {url}:\n{content}")]

async def main():
    await mcp.run_async(
        "streamable-http",
        host="0.0.0.0",
        port=8085,
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())