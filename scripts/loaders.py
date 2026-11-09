from pathlib import Path
from dataclasses import dataclass, field
from pypdf import PdfReader
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

import docx

NOISE = ["script", "style", "nav", "footer", "header", "aside"]

@dataclass 
class Document:
    content: str
    metadata: dict =  field(default_factory=dict)

    def __repr__(self):
        preview = " ".join(self.content.split())[:60]
        return f"Document({self.metadata}, content={preview!r}…)"

def _load_pdf(path: str) -> list[Document]:
    """PDF → text, page by page (pypdf).

    A PDF isn't linear text: it's a layout of blocks. `pypdf` extracts the text of
    each page; we keep the page number in the metadata (useful for citations and
    retrieval debugging). It copes with linear text; tables/columns/scans break it
    → those need managed parsers (see Module 2 · Lesson 6, LlamaParse).
    """
    reader = PdfReader(path)
    docs = []

    for i, page in enumerate(reader.pages):
        content = page.extract_text() or ""
        docs.append(Document(content, {"source": path, "page": i + 1, "type": "pdf"}))

    return docs

def _load_docx(path: str) -> list[Document]:
    """DOCX → text from the XML paragraphs (python-docx). Usually a single Document.

    A .docx is a zip archive of XML: the text is there, but buried in tags. `python-docx`
    exposes it to us as a list of paragraphs.

    Note: `import docx` is the installed **python-docx** package (the import name
    differs from the package name). Don't add a `docx.py` to this folder: these
    scripts import each other by bare module name, so it would shadow the package.
    """
    document = docx.Document(path)
    content = "\n".join(p.text for p in document.paragraphs if p.text)
    return [Document(content, {"source": path, "type": "docx"})]

def _load_html(path: str) -> list[Document]:
    """HTML → text (BeautifulSoup), stripping the most obvious noise first.

    A web page is a tree of tags where the useful content is surrounded by menus,
    footers, banners, scripts. If we take all the text, the *noise* ends up in the
    prompt and confuses the model: its answers end up quoting the menu.

    Here we remove the noisiest tags — the bare minimum. "Cleaning properly" is a
    craft of its own: that's the next lesson (Module 2 · Lesson 5, Text Cleaning).
    """
    with open(path, encoding="utf-8") as f:
        parsed_html = BeautifulSoup(f, "html.parser") 

    for tag in parsed_html(NOISE):
        tag.decompose()

    content = parsed_html.get_text(separator="\n", strip=True)
    return [Document(content, {"source": path, "type": "html" })]

def _load_markdown(path: str) -> list[Document]:
    """Markdown → text (markdown-it-py).

    The friendliest format: it's already text + a lightweight syntax. We parse it (render →
    HTML → text) to handle the syntax cleanly instead of leaving it raw.
    """
    with open(path, encoding="utf-8") as f:
        source = f.read()

    _md = MarkdownIt()

    html = _md.render(source)
    content = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)

    return [Document(content, {"source": path, "type": "markdown"})]

def _load_txt(path: str) -> list[Document]:
    """TXT → text (direct read).

    The simplest case: a text file is already text. No library, just
    `read()`. We keep it because in Data Loading (Module 2 · Lesson 4) the `.txt` was the students'
    exercise, and in Module 2 · Lesson 5 we need a "raw but clean, just needs reformatting" case
    (the Project Gutenberg book: boilerplate + lines wrapped at ~70 characters).
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    return [Document(content, {"source": path, "type": "txt"})]

def load(path: str) -> list[Document]:
    """load(): a single entry point.

    Whoever loads a document shouldn't need to know which library handles which file. A
    single function dispatches by extension and always returns `Document`s.
    Adding a format = adding a `_load_<format>()` function in this file and a branch here.
    """
    extension = Path(path).suffix.lower()

    if extension == ".pdf":
        return _load_pdf(path)
    if extension == ".docx":
        return _load_docx(path)
    if extension in (".htm", ".html"):
        return _load_html(path)
    if extension in (".md", ".markdown"):
        return _load_markdown(path)
    if extension == ".txt":
        return _load_txt(path)
    
    raise ValueError(f"\t{path.stem}: unsupported document format '{extension}'")
