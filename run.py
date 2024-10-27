import os

from dotenv import load_dotenv

from bot.elf_builder import ELF

print("Instantiating ELF...")
bot = ELF()

cogs_list = [
    'general',
    'database',
    'elfvision'
]

print("Loading extensions...")

for cog in cogs_list:
    bot.load_extension(f"commands.{cog}")
    print(f"Loaded extension: {cog}")

print("Running ELF...")
load_dotenv()
bot.run(os.getenv("TOKEN"))