class MovideskArticle:
    """Representa um artigo do Movidesk."""
    def __init__(self, title: str, content: str, url: str):
        self.title = title
        self.content = content
        self.url = url

    @property
    def is_valid(self) -> bool:
        """Valida se o artigo possui título e conteúdo."""
        return bool(self.title and self.content)

def sanitize_content(content: str) -> str:
    """Sanitiza o conteúdo do artigo removendo caracteres indesejados."""
    return content.strip()

def format_movidesk_url(article_id: int, slug: str) -> str:
    """Formata a URL do artigo do Movidesk."""
    base_url = "https://sisand.movidesk.com/kb/pt-br/article"
    return f"{base_url}/{article_id}/{slug}"
