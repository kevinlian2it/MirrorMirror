from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from app.utils import get_video_id
import logging
import re


class TranscriptService:
    def __init__(self):
        self.formatter = TextFormatter()

    def remove_emojis(self, text):
        """Removes emojis and special characters from text."""
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # Emoticons
            u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # Transport & map symbols
            u"\U0001F700-\U0001F77F"  # Alchemical symbols
            u"\U0001F780-\U0001F7FF"  # Geometric shapes
            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess symbols, objects
            u"\U0001FA70-\U0001FAFF"  # More symbols
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)

    def get_transcript(self, video_url: str) -> str:
        """Fetches and formats transcript for a given YouTube video URL."""
        try:
            video_id = get_video_id(video_url)
            if not video_id:
                logging.error(f"Could not extract video ID from: {video_url}")
                return None

            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            formatted_transcript = self.formatter.format_transcript(transcript)
            formatted_transcript = self.remove_emojis(formatted_transcript)
            formatted_transcript = re.sub(r'\s+', ' ', formatted_transcript).strip()
            #logging.error(formatted_transcript)
            return formatted_transcript

        except Exception as e:
            logging.error(f"Error downloading transcript for {video_url}: {e}")
            return None
