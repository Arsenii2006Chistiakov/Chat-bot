#!/usr/bin/env python3

"""check point wiht the bot being able to query the database from nl"""


"""
MongoDB-Connected Chatbot
A CLI chatbot that uses an LLM to translate natural language into
MongoDB queries and displays the results.

Dependencies:
- pymongo
- python-dotenv
- rich
- openai (used for the LLM API call)

You will need to have a MongoDB instance running and
set the OPENAI_API_KEY and MONGO_URI environment variables.
"""

import os
import json
import sys
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import pymongo
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.syntax import Syntax
import json

# Initialize Rich console
console = Console()

class MongoDBQueryGenerator:
    """
    A class to generate MongoDB query JSON from natural language using an LLM.
    """
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            console.print(Panel(
                "[red]Error: OPENAI_API_KEY not found in environment variables![/red]",
                title="Configuration Error", border_style="red"
            ))
            sys.exit(1)
        
        # Load example document
        try:
            with open('example_doc.json', 'r') as f:
                self.example_doc = json.load(f)
        except FileNotFoundError:
            console.print(Panel(
                "[red]Error: example_doc.json not found![/red]",
                title="Configuration Error", border_style="red"
            ))
            sys.exit(1)
            
    async def get_query_from_llm(self, natural_language_query: str, collection_schema: Dict[str, Any]) -> str:
        """
        Sends the user's query and the schema to an LLM to get a structured MongoDB query.
        
        Args:
            natural_language_query (str): The user's question.
            collection_schema (Dict[str, Any]): A JSON-like representation of the collection schema.
            
        Returns:
            str: A JSON string of the MongoDB query.
        """
        prompt = f"""
You are an expert at converting natural language questions into structured MongoDB queries.
You will be provided with an example document from the collection and a user's question.
Your task is to generate a JSON object that can be used directly in a PyMongo find() operation.
The JSON object should have the following keys:
- "filter": A MongoDB query filter (e.g., {{ "artist": "Capone" }}).
- "projection": A MongoDB projection to select specific fields (e.g., {{ "title": 1, "artist": 1, "views": 1, "_id": 0 }}).
- "sort": A MongoDB sort object (e.g., {{ "views": -1 }}).
- "limit": An integer limit for the number of results.

        Here is an example document from the collection:
        {json.dumps(self.example_doc, indent=2)}

---
Here are some examples of natural language queries and the correct JSON output:

User: "Find all songs that were ever ranked #1 in Brazil"
{{
  "filter": {{ "charts.Brazil.rank": 1 }},
  "projection": {{ "song_name": 1, "artist_name": 1, "charts.Brazil": 1, "_id": 0 }},
  "sort": {{}},
  "limit": 10
}}

User: "List the top 5 songs by Ponte Perro that charted in Argentina"
{{
  "filter": {{
    "artist_name": "Ponte Perro",
    "charts.Argentina.rank": {{ "$lte": 5 }}
  }},
  "projection": {{ "song_name": 1, "artist_name": 1, "charts.Argentina": 1, "_id": 0 }},
  "sort": {{ "charts.Argentina.rank": 1 }},
  "limit": 5
}}

User: "Show me all Latin songs with Spanish lyrics"
{{
  "filter": {{ "genres": "Latin", "language_code": "spa" }},
  "projection": {{ "song_name": 1, "artist_name": 1, "genres": 1, "language_code": 1, "_id": 0 }},
  "sort": {{ "first_seen": -1 }},
  "limit": 10
}}

User: "Find songs that are longer than 3 minutes and have high language probability"
{{
  "filter": {{ 
    "audio_metadata.duration": {{ "$gt": 180 }},
    "language_probability": {{ "$gt": 0.8 }}
  }},
  "projection": {{ "song_name": 1, "artist_name": 1, "audio_metadata.duration": 1, "language_probability": 1, "_id": 0 }},
  "sort": {{ "language_probability": -1 }},
  "limit": 10
}}

User: "Get all songs with RKT genre that are still being processed"
{{
  "filter": {{ "genres": "RKT", "TREND_STATUS": "UNPROCESSED" }},
  "projection": {{ "song_name": 1, "artist_name": 1, "genres": 1, "TREND_STATUS": 1, "_id": 0 }},
  "sort": {{ "first_seen": -1 }},
  "limit": 10
}}

User: "{natural_language_query}"

Provide only the JSON output. Do not include any other text or explanation.
"""
        
        import openai
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=self.api_key)
        
        try:
            # Make API call to OpenAI
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are an expert at converting natural language questions into structured MongoDB queries. Provide only JSON output."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            # Extract the response text
            text = response.choices[0].message.content.strip()
            print(text)
            return text
            
        except Exception as e:
            console.print(f"[red]Debug - OpenAI API call failed: {e}[/red]")
            return "{}"

class MongoChatbot:
    """
    Main chatbot class to handle user interaction and database operations.
    """
    def __init__(self):
        load_dotenv()
        self.mongo_uri = os.getenv('MONGO_URI')
        if not self.mongo_uri:
            console.print(Panel(
                "[red]Error: MONGO_URI not found in environment variables![/red]",
                title="Configuration Error", border_style="red"
            ))
            sys.exit(1)

        try:
            self.client = pymongo.MongoClient(self.mongo_uri)
            self.db = self.client.HUGO_MODEL_DB
            self.collection = self.db.TOP_TIKTOK_SOUNDS
            self.query_generator = MongoDBQueryGenerator()
        except pymongo.errors.ConnectionFailure as e:
            console.print(Panel(
                f"[red]Error connecting to MongoDB: {e}[/red]",
                title="Connection Error", border_style="red"
            ))
            sys.exit(1)
        
        # Updated schema to reflect the new document structure with charts
        self.collection_schema = {
            "_id": "ObjectId",
            "song_id": "string",
            "artist_name": "string",
            "charts": {
                "country_code_example": [{
                    "timestamp": "string (date)",
                    "rank": "number"
                }]
            },
            "first_seen": "string (date)",
            "song_name": "string",
            "sound_link": "string (URL)",
            "genres": ["string"],
            "gcs_path": "string",
            "status": "string",
            "embedding_path": "string",
            "transcription_status": "string",
            "audio_metadata": {
                "format_name": "string",
                "duration": "number",
                "bit_rate": "number",
                "size": "number",
                "codec_name": "string",
                "sample_rate": "number",
                "channels": "number"
            },
            "language_code": "string",
            "language_probability": "number",
            "lyrics": "string",
            "music_embedding": ["number"],
            "TREND_STATUS": "string"
        }

    def _format_results(self, documents: List[Dict[str, Any]]):
        """Formats the list of documents for a nice display."""
        if not documents:
            console.print("[yellow]No documents found matching your query.[/yellow]")
            return

        for i, doc in enumerate(documents, 1):
            song_name = doc.get("song_name", "N/A")
            artist_name = doc.get("artist_name", "N/A")
            genres = ', '.join(doc.get("genres", []))
            
            charts_info = ""
            charts = doc.get("charts", {})
            for country, chart_entries in charts.items():
                if chart_entries:
                    latest_entry = chart_entries[0]  # Assuming sorted by timestamp
                    rank_data = latest_entry.get("rank", {})
                    # Handle both formats: {"$numberInt": "1"} and direct integer
                    if isinstance(rank_data, dict) and "$numberInt" in rank_data:
                        rank = rank_data["$numberInt"]
                    elif isinstance(rank_data, (int, str)):
                        rank = str(rank_data)
                    else:
                        rank = "N/A"
                    timestamp = latest_entry.get("timestamp", "N/A")
                    charts_info += f"  - [bold]{country}:[/bold] Rank {rank} (as of {timestamp})\n"

            panel_content = (
                f"[bold blue]Result {i}[/bold blue]\n\n"
                f"[bold]Song Name:[/bold] {song_name}\n"
                f"[bold]Artist:[/bold] {artist_name}\n"
                f"[bold]Genres:[/bold] {genres}\n"
                f"[bold]First Seen:[/bold] {doc.get('first_seen', 'N/A')}\n"
            )
            
            if charts_info:
                panel_content += f"\n[bold]Latest Chart Ranks:[/bold]\n{charts_info}"

            console.print(Panel(
                panel_content,
                border_style="green"
            ))

    async def run(self):
        """Main application loop."""
        console.print(Panel(
            "[bold blue]🤖 MongoDB Song Database Chatbot CLI[/bold blue]\n\n"
            "Ask me questions about the song database!\n\n"
            "Example questions:\n"
            "• Find all songs that were ranked #1 in Brazil.\n"
            "• Show me songs by Ponte Perro that charted in Argentina.\n"
            "• List all Latin songs with Spanish lyrics.\n"
            "• Find songs longer than 3 minutes with high language probability.\n\n"
            "[yellow]Type 'quit' or 'exit' to end the session.[/yellow]",
            title="Welcome", border_style="blue"
        ))

        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
                if user_input.lower() in ['quit', 'exit']:
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    break
                
                if not user_input:
                    continue

                # Use a typing indicator while the LLM and DB are working
                with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                    # 1. Get the structured query from the LLM
                    query_json_str = await self.query_generator.get_query_from_llm(user_input, self.collection_schema)
                    
                    try:
                        query_dict = json.loads(query_json_str)
                    except json.JSONDecodeError:
                        console.print(Panel(
                            f"[red]Error parsing LLM response. Got: {query_json_str}[/red]",
                            title="LLM Error", border_style="red"
                        ))
                        continue

                    # 2. Execute the query
                    filter_criteria = query_dict.get('filter', {})
                    projection = query_dict.get('projection', {})
                    sort = query_dict.get('sort', None)
                    limit = query_dict.get('limit', 10) # Default limit

                    # Check for empty filter to prevent full collection scans
                    if not filter_criteria or filter_criteria == {}:
                         console.print(Panel(
                            "[yellow]Your query was too broad. Please be more specific![/yellow]",
                            title="Query Error", border_style="yellow"
                        ))
                         continue

                    # Execute the find operation
                    cursor = self.collection.find(filter_criteria, projection)
                    if sort:
                        cursor = cursor.sort(sort)
                    if limit:
                        cursor = cursor.limit(limit)

                    results = list(cursor)

                # 3. Display the formatted results
                console.print()
                self._format_results(results)
                console.print()
            
            except Exception as e:
                console.print(f"[red]An unexpected error occurred: {e}[/red]")

async def main():
    """Entry point of the application."""
    try:
        chatbot = MongoChatbot()
        await chatbot.run()
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())