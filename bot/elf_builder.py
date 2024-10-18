import discord

class ELF(discord.Bot):
    async def on_ready(self):
        print(f"Logged in as {self.user.name}")