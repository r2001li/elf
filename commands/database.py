import discord
from discord.ext import commands

import sqlite3

from config import config

class Database(commands.Cog):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot

        print(f"Connecting to database: {config.DATABASE_FILENAME}")
        self.conn = sqlite3.connect(config.DATABASE_DIR + "/" + config.DATABASE_FILENAME)

        cursor = self.conn.cursor()

        # Create a relation for tags (if not done already)
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS Tags 
                       (
                       guildID integer,
                       tagName text,
                       tagContent text,
                       PRIMARY KEY (guildID, tagName)
                       )
                       ''')

        self.conn.commit()
        cursor.close()
    
    ### Tags Command Group ###
    tag = discord.SlashCommandGroup(name="tag", description="Manage tags")
    
    # Add tag
    @tag.command()
    async def add(self, ctx, tag, content):
        cursor = self.conn.cursor()
        guildID = ctx.guild.id

        # Check if tag already exists for this server
        cursor.execute(f"SELECT tagContent FROM Tags WHERE guildID = {guildID} AND tagName = \'{tag}\'")
        res = cursor.fetchone()

        if not res is None:
            self.conn.commit()
            cursor.close()
            return await ctx.respond(f"Tag \'{tag}\' already exists in this server.")
        
        # Add new tag to database
        cursor.execute(f"INSERT INTO Tags VALUES ({guildID}, \'{tag}\', \'{content}\')")
        self.conn.commit()
        cursor.close()

        await ctx.respond(f"Added tag: {tag}")
    
    # Fetch tag
    @tag.command()
    async def get(self, ctx, tag):
        cursor = self.conn.cursor()
        guildID = ctx.guild.id
        cursor.execute(f"SELECT tagContent FROM Tags WHERE guildID = {guildID} AND tagName = \'{tag}\'")
        res = cursor.fetchone()

        if res is None:
            self.conn.commit()
            cursor.close()
            await ctx.respond(f"No matching tag found for: {tag}")
            return
        
        content = res[0]
        self.conn.commit()
        cursor.close()

        await ctx.respond(f"tag: {tag}\n{content}")
    
    # List all tags
    @tag.command()
    async def list(self, ctx):
        cursor = self.conn.cursor()
        guildID = ctx.guild.id

        cursor.execute(f"SELECT tagName FROM Tags WHERE guildID = {guildID}")
        res = cursor.fetchall()

        if not res:
            self.conn.commit()
            cursor.close()
            await ctx.respond(f"No tags found for the current server.")
            return

        self.conn.commit()
        cursor.close()

        tags = map(lambda x: x[0], res)
        await ctx.respond("\n".join(tags))

    
def setup(bot):
    bot.add_cog(Database(bot)) 