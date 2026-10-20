"""Pillow contact sheets for fixed archived SLAKE samples; CPU only."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);root=p.parse_args().root
    rows=json.loads((root/'slake_sample_60.json').read_text(encoding='utf-8'))['rows']
    font_path='C:/Windows/Fonts/msyh.ttc'
    f=ImageFont.truetype(font_path,18)
    for start in range(0,len(rows),10):
        sheet=Image.new('RGB',(2000,1900),'white');d=ImageDraw.Draw(sheet)
        for j,row in enumerate(rows[start:start+10]):
            x=(j%2)*1000;y=(j//2)*380
            im=Image.open(root/row['exported_image']).convert('RGB');im.thumbnail((330,325))
            sheet.paste(im,(x+8,y+35))
            d.text((x+8,y+4),f"#{row['review_index']}  qid={row['question_id']} {row['category']}",font=f,fill='black')
            text=f"{row['question_type']} / {row['answer_type']} / {row['language']}\nQ: {row['question']}\nGT: {row['reference']}\nV1: {row['v1_prediction']}\nr2: {row['r2_prediction']}\nImage: {row['image']}"
            lines=[]
            for para in text.splitlines():
                current=''
                for char in para:
                    if d.textlength(current+char,font=f)>640:
                        lines.append(current);current=char
                    else:current+=char
                lines.append(current)
            d.multiline_text((x+350,y+38),'\n'.join(lines),font=f,fill='black',spacing=6)
        sheet.save(root/f'review_sheet_{start//10+1}.png')

if __name__=='__main__':main()
