#!/usr/bin/env python3
"""
CLI client for the MongoDB Chatbot API (app.py).

Type prompts and commands like in query_gpt.py, but the client sends
requests to the FastAPI server so multiple users can use it concurrently.

Environment variables:
- CHAT_API_URL (default: http://localhost:8000)
- USER_ID (default: cli_user)
"""

import os
import sys
import requests
from typing import Dict, Any, List
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt


console = Console()


def print_welcome(api_url: str, user_id: str) -> None:
    console.print(Panel(
        """
[bold blue]🤖 MongoDB Song Database Chatbot (HTTP Client)[/bold blue]

Type your questions or use commands:
• /load /path/to/file.mp3
• /loadtext --text "lyrics here"  (or just: /loadtext your lyrics)
• /search --file /path/to/file.mp3 --prompt "filters here"
• /search --prompt "Latin songs in Brazil"
• /history  |  /clear  |  /add <index>

Type 'quit' or 'exit' to end the session.
""".strip(),
        title=f"Connected to {api_url} as {user_id}", border_style="blue"
    ))


def print_search_results(results: List[Dict[str, Any]]) -> None:
    if not results:
        console.print(Panel("No results.", border_style="yellow"))
        return
    console.print(Panel(f"Found {len(results)} results", title="Search Results", border_style="green"))
    for i, result in enumerate(results, 1):
        song_name = result.get("song_name", "N/A")
        artist_name = result.get("artist_name", "N/A")
        genres = ", ".join(result.get("genres", []))
        trend_description = result.get("trend_description")
        trend_status = result.get("TREND_STATUS")
        trend_info = f"[bold]Trend:[/bold] {trend_description}\n" if trend_description and trend_status == "PROCESSED" else ""
        lyrics_preview = (result.get("lyrics", "") or "N/A")
        if isinstance(lyrics_preview, str) and len(lyrics_preview) > 120:
            lyrics_preview = lyrics_preview[:120] + "..."

        content = (
            f"[bold blue]Result {i}[/bold blue]\n\n"
            f"[bold]Song Name:[/bold] {song_name}\n"
            f"[bold]Artist:[/bold] {artist_name}\n"
            f"[bold]Genres:[/bold] {genres}\n"
            f"{trend_info}"
            f"[bold]Lyrics Preview:[/bold] {lyrics_preview}\n"
        )
        console.print(Panel(content, border_style="green"))


def get_context(api_url: str, user_id: str) -> None:
    try:
        r = requests.get(f"{api_url}/context/{user_id}", timeout=30)
        if r.status_code != 200:
            console.print(Panel(f"Error getting context: {r.text}", border_style="red"))
            return
        data = r.json()
        items = data.get("context_songs", [])
        if not items:
            console.print(Panel("No context items.", title="Context", border_style="yellow"))
            return
        console.print(Panel(f"{len(items)} context items", title="Context", border_style="cyan"))
        for i, item in enumerate(items, 1):
            title = item.get("song_name") or item.get("title") or "N/A"
            artist = item.get("artist_name") or item.get("artist") or "N/A"
            console.print(f"{i}. {title} by {artist}")
    except Exception as e:
        console.print(Panel(f"Context error: {e}", border_style="red"))


def main() -> None:
    api_url = os.getenv("CHAT_API_URL", "http://localhost:8000").rstrip("/")
    user_id = os.getenv("USER_ID", "cli_user")

    print_welcome(api_url, user_id)

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit"}:
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break

            # Optional direct context fetch for convenience
            if user_input.lower() == "/context":
                get_context(api_url, user_id)
                continue

            payload = {"message": user_input, "user_id": user_id}
            resp = requests.post(f"{api_url}/chat", json=payload, timeout=180)
            if resp.status_code != 200:
                console.print(Panel(f"[red]API Error {resp.status_code}[/red]\n{resp.text}", title="Error", border_style="red"))
                continue

            result = resp.json()
            response_text = result.get("response", "")
            category = result.get("category", "talk")
            search_results = result.get("search_results")

            # Show category header like the CLI experience
            console.print(f"[bold magenta]🎯 Category: {category.upper()}[/bold magenta]")

            # Main response panel
            console.print(Panel(response_text, title="Assistant", border_style="green"))

            # If search results are present, show nicely
            if search_results:
                print_search_results(search_results)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Bye![/yellow]")
            break
        except Exception as e:
            console.print(Panel(f"Unexpected error: {e}", border_style="red"))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(Panel(f"Fatal error: {e}", border_style="red"))
        sys.exit(1)


