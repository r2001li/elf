import discord
from discord.ext import commands
from discord.commands import Option
import datetime

class General(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    
    @commands.slash_command(description="View an account's date of creation and when it joined this server.")
    async def userage(self, ctx, user: Option(discord.Member)):
        date_format = "%a, %d %b %Y %I:%M %p"

        embed = discord.Embed(description=user.mention)
        embed.set_author(name=str(user), icon_url=user.display_avatar.url)
        embed.set_thumbnail(url=user.display_avatar.url)
        embed.add_field(name="Joined", value=user.joined_at.strftime(date_format))
        embed.add_field(name="Registered", value=user.created_at.strftime(date_format))

        return await ctx.respond(embed=embed)

def setup(bot):
    bot.add_cog(General(bot))       