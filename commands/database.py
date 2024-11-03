import discord
from discord.ext import commands

import sqlite3
import aiosqlite

from config import config

DATABASE_PATH = config.DATABASE_DIR + "/" + config.DATABASE_FILENAME

class Database(commands.Cog):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot

        print(f"Connecting to database: {config.DATABASE_FILENAME}")
        with sqlite3.connect(DATABASE_PATH) as conn:
            # Create a relation for tags (if not done already)
            conn.execute('''
                        CREATE TABLE IF NOT EXISTS Tags 
                        (
                        guildID integer,
                        tagName text,
                        tagContent text,
                        PRIMARY KEY (guildID, tagName)
                        )
                        ''')
            conn.commit()
    
    ### Tags Command Group ###
    tag = discord.SlashCommandGroup(name="tag", description="Manage tags")
    
    # Add tag
    @tag.command()
    async def add(self, ctx, tag, content):
        guildID = ctx.guild.id
        
        # Add new tag to database
        async with aiosqlite.connect(DATABASE_PATH) as conn:    
            await conn.execute("INSERT INTO Tags VALUES (?, ?, ?)", (guildID, tag, content))
            await conn.commit()

        return await ctx.respond(f"Added tag: {tag}")
    
    # Fetch tag
    @tag.command()
    async def get(self, ctx, tag):
        guildID = ctx.guild.id

        async with aiosqlite.connect(DATABASE_PATH) as conn: 
            cursor = await conn.execute(f"SELECT tagContent FROM Tags WHERE guildID = ? AND tagName = ?", (guildID, tag))
            res = await cursor.fetchone()

        if not res:
            return await ctx.respond(f"No matching tag found for: {tag}")
        
        content = res[0]
        return await ctx.respond(f"Tag: {tag}\n{content}")
    
    # List all tags
    @tag.command()
    async def list(self, ctx):
        guildID = ctx.guild.id

        async with aiosqlite.connect(DATABASE_PATH) as conn: 
            cursor = await conn.execute("SELECT tagName FROM Tags WHERE guildID = ?", (guildID,))
            res = await cursor.fetchall()
        
        if not res:
            return await ctx.respond(f"No tags found for the current server.")

        tags = map(lambda x: x[0], res)
        await ctx.respond("\n".join(tags))

    
def setup(bot):
    bot.add_cog(Database(bot)) 