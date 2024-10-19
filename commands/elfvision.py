import discord
from discord.ext import commands

import torch

from pathlib import Path

from config import config
from vision import engine
from vision import data_handler

class ELFVision(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = data_handler.get_classes()

        # Load saved model
        target_dir = Path(config.MODEL_DIR)
        model_path = target_dir / config.MODEL_NAME
        
        try:
            self.model = torch.load(f=model_path, weights_only=False)
        except FileNotFoundError or FileExistsError:
            print(f"Model file not fount at {target_dir}")
    
    @discord.slash_command()
    async def guess(self, ctx, img: discord.Attachment):
        '''
        if img.content_type != "image":
            await ctx.respond("Please provide an image.")
            return
        '''
        
        file = await img.to_file()
        label, confidence = engine.predict_image(model=self.model, 
                                                       class_names=self.class_names, 
                                                       image_path=file.fp, 
                                                       device=self.device)
        
        # await ctx.respond(f"Content: {label}\nConfidence: {confidence * 100:.2f}%", file=file)
        await ctx.respond(embed=discord.Embed(image=img.url, description=f"Content: {label}\nConfidence: {confidence * 100:.2f}%"))

def setup(bot):
    bot.add_cog(ELFVision(bot))    
