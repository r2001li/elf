import discord
from discord.ext import commands

import sqlite3

from config import config

class Database(commands.Cog):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot

        self.conn = sqlite3.connect(config.DATABASE_DIR + "/" + config.DATABASE_FILENAME)

        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS tags (guildID integer, tagName text UNIQUE PRIMARY KEY, tagContent text)''')
        self.conn.commit()
        cursor.close()
    
    @discord.slash_command()
    async def tag_add(self, ctx, tag, content):
        cursor = self.conn.cursor()
        guildID = ctx.guild.id
        cursor.execute(f"INSERT INTO tags VALUES ({guildID}, \'{tag}\', \'{content}\')")
        self.conn.commit()
        cursor.close()

        await ctx.respond(f"Added tag \'{tag}\'.")
    
    @discord.slash_command()
    async def tag_get(self, ctx, tag):
        cursor = self.conn.cursor()
        guildID = ctx.guild.id
        cursor.execute(f"SELECT tagContent FROM tags WHERE guildID = {guildID} AND tagName = \'{tag}\'")
        res = cursor.fetchone()

        if res is None:
            self.conn.commit()
            cursor.close()
            await ctx.respond(f"No matching tag found for \'{tag}\'.")
            return
        
        content = res[0]
        self.conn.commit()
        cursor.close()

        await ctx.respond(content)

    
def setup(bot):
    bot.add_cog(Database(bot)) 