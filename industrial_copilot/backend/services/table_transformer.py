import json
from openai import OpenAI
from unified_rag.config import settings

class TableTransformer:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)

    def summarize_table(self, table_json: str, context: str = "") -> str:
        """
        Uses an LLM to transform raw technical table JSON into a dense, searchable summary.
        """
        prompt = (
            "You are a Technical Data Specialist. Convert this raw table JSON into a concise, searchable summary.\n"
            f"Context: {context}\n"
            "Format your response as a clear description. Focus on key specifications, ranges, and part numbers.\n"
            "Include every unique column name and its meaning in the context of the technical manual.\n"
            "If it's a troubleshooting table, list the Problem-Cause-Solution pairs in a dense format."
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You convert structured technical data into dense, searchable text summaries."},
                    {"role": "user", "content": f"{prompt}\n\nRAW TABLE DATA:\n{table_json}"}
                ],
                max_tokens=400,
                temperature=0.0
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            print(f"❌ [TableTransformer] Error summarizing table: {e}")
            return f"Table Data: {table_json[:500]}..."
