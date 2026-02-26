"""
Low-level Telegram Bot API client.
Uses requests (already in deps) — no extra libraries needed.
"""
import logging
import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = 'https://api.telegram.org/bot{token}/{method}'
MAX_MESSAGE_LENGTH = 4096


class TelegramBot:
    def __init__(self, token):
        self.token = token

    def _call(self, method, **params):
        url = TELEGRAM_API.format(token=self.token, method=method)
        resp = requests.post(url, json=params, timeout=30)
        data = resp.json()
        if not data.get('ok'):
            logger.error(f"Telegram API error: {method} → {data}")
        return data

    def send_message(self, chat_id, text, parse_mode='Markdown'):
        """Send a text message. Auto-chunks if >4096 chars."""
        chunks = self._chunk_text(text)
        results = []
        for chunk in chunks:
            result = self._call(
                'sendMessage',
                chat_id=chat_id,
                text=chunk,
                parse_mode=parse_mode,
                disable_web_page_preview=True,
            )
            results.append(result)
        return results[-1] if results else None

    def send_photo(self, chat_id, photo_path, caption=None):
        """Send a photo via multipart upload."""
        url = TELEGRAM_API.format(token=self.token, method='sendPhoto')
        data = {'chat_id': chat_id}
        if caption:
            data['caption'] = caption[:1024]
            data['parse_mode'] = 'Markdown'
        with open(photo_path, 'rb') as f:
            resp = requests.post(url, data=data, files={'photo': f}, timeout=60)
        result = resp.json()
        if not result.get('ok'):
            logger.error(f"Telegram sendPhoto error: {result}")
        return result

    def set_webhook(self, url):
        return self._call('setWebhook', url=url)

    def delete_webhook(self):
        return self._call('deleteWebhook')

    def get_webhook_info(self):
        return self._call('getWebhookInfo')

    @staticmethod
    def parse_update(data):
        """Extract (chat_id, user_id, text, username) from an Update dict."""
        msg = data.get('message') or data.get('edited_message') or {}
        chat_id = msg.get('chat', {}).get('id')
        user = msg.get('from', {})
        user_id = user.get('id')
        username = user.get('username', '')
        text = msg.get('text', '')
        return chat_id, user_id, text, username

    @staticmethod
    def _chunk_text(text):
        """Split text into chunks of MAX_MESSAGE_LENGTH, breaking at newlines."""
        if len(text) <= MAX_MESSAGE_LENGTH:
            return [text]
        chunks = []
        while text:
            if len(text) <= MAX_MESSAGE_LENGTH:
                chunks.append(text)
                break
            # Find last newline before limit
            cut = text.rfind('\n', 0, MAX_MESSAGE_LENGTH)
            if cut == -1:
                cut = MAX_MESSAGE_LENGTH
            chunks.append(text[:cut])
            text = text[cut:].lstrip('\n')
        return chunks
