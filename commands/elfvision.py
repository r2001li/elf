import discord
from discord.ext import commands

import torch

from safetensors.torch import load_model

from pathlib import Path
from PIL import UnidentifiedImageError

from config import config
from vision import model_builder
from vision import engine
from vision import data_handler

class ELFVision(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.class_names = data_handler.get_classes()
        self.model = model_builder.ELFVision2Model(input_shape=config.INPUT_SHAPE, output_shape=len(self.class_names))

        # Load saved model
        model_dir = Path(config.MODEL_DIR)
        model_path = model_dir / config.MODEL_NAME

        print("Loading model state dict")
        load_model(model=self.model, filename=model_path, device=self.device)
        
        '''
        try:
            self.model = torch.load(f=model_path, weights_only=False)
        except FileNotFoundError or FileExistsError:
            print(f"Model file not fount at {model_dir}")
        '''
    
    @discord.slash_command(description="Sends an image for ELF to guess.")
    async def guess(self, ctx, image: discord.Attachment):
        file = await image.to_file()

        try:
            label, confidence = engine.predict_image(model=self.model, 
                                                       class_names=self.class_names, 
                                                       image_path=file.fp, 
                                                       device=self.device)
        except (UnidentifiedImageError, FileNotFoundError):
            await ctx.respond("Invalid format. Please provide a valid image file.")
            return
        
        embed = discord.Embed()
        embed.set_author(name=str(ctx.author), icon_url=ctx.author.display_avatar.url)
        embed.set_image(url=image.url)
        embed.add_field(name="Content", value=label)
        embed.add_field(name="Confidence", value=f"{confidence * 100:.2f}%")
        
        # await ctx.respond(embed=discord.Embed(image=image.url, description=f"Content: {label}\nConfidence: {confidence * 100:.2f}%"))

        await ctx.respond(embed=embed)

def setup(bot):
    bot.add_cog(ELFVision(bot))    
