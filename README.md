# PsychoNewBot

Telegram bot platform with AI-powered dialogues, subscriptions, media features and a companion MAX messenger bot.

## Requirements

- Python 3.12+
- A Telegram bot token
- Database connection settings

## Local setup

```bash
cp config.ini.example config.ini
```

Fill in `config.ini` locally. This file contains runtime credentials and is intentionally excluded from Git. Install the project's Python dependencies in your virtual environment before starting the bot.

## Run

```bash
python main.py
```

## Checks

```bash
pytest
```
