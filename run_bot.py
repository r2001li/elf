import os

from dotenv import load_dotenv

from config import config
from bot.elf_builder import ELF

print("Instantiating ELF...")
bot = ELF()

print("Loading extensions...")

for cog in config.COGS_LIST:
    bot.load_extension(f"commands.{cog}")
    print(f"Loaded extension: {cog}")

print("Running ELF...")
load_dotenv()
bot.run(os.getenv("TOKEN"))
