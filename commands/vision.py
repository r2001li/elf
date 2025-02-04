import discord
from discord.ext import commands

import torch

from safetensors.torch import load_model

from PIL import UnidentifiedImageError

from vision import elfvision
from vision import vision_engine
from vision import utils

class Vision(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

        self.device = utils.get_device()
        print(f"Using device: {self.device}")

        self.class_names = utils.get_classes()
        self.model = elfvision.ELFVisionNN(output_shape=len(self.class_names))

        # Load saved model
        print("Loading vision model...")

        if utils.model_exists():
            model_path = utils.get_model_path()
            load_model(model=self.model, filename=model_path, device=self.device)
            self.has_model = True
        else:
            print("No vision model found. Computer vision functions will be disabled.")
            self.has_model = False
    
    @discord.slash_command(description="Sends an image for ELF to guess.")
    async def guess(self, ctx, image: discord.Attachment):
        if not self.has_model:
            return await ctx.respond("Computer vision functions are disabled.")

        file = await image.to_file()

        try:
            label, confidence = vision_engine.predict_image(model=self.model, 
                                                       class_names=self.class_names, 
                                                       image_path=file.fp, 
                                                       device=self.device)
        except (UnidentifiedImageError, FileNotFoundError):
            return await ctx.respond("Invalid format. Please provide a valid image.")
            
        embed = discord.Embed()
        embed.set_author(name=str(ctx.author), icon_url=ctx.author.display_avatar.url)
        embed.set_image(url=image.url)
        embed.add_field(name="Content", value=label)
        embed.add_field(name="Confidence", value=f"{confidence * 100:.2f}%")

        return await ctx.respond(embed=embed)

def setup(bot):
    bot.add_cog(Vision(bot))    
