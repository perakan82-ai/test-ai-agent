---
name: reach
description: อ่านเนื้อหาจากลิงก์ที่ผู้ใช้ส่งมา - หน้าเว็บ บทความ คลิป YouTube repo GitHub และฟีด RSS ใช้ทุกครั้งที่ผู้ใช้ส่งลิงก์แล้วขอให้สรุป อ่าน หรือวิเคราะห์
---

# Reach

ใช้สคริปต์ `reach.py` ในโฟลเดอร์เดียวกับไฟล์นี้ (`.claude/skills/reach/reach.py`)

## เลือกคำสั่งตามลิงก์
| ลิงก์ | คำสั่ง |
|---|---|
| youtube.com / youtu.be | `python .claude/skills/reach/reach.py yt <url>` |
| github.com/<owner>/<repo> | `python .claude/skills/reach/reach.py gh <owner/repo>` |
| ลิงก์ที่ลงท้าย .xml /rss /feed หรือผู้ใช้บอกว่าเป็นฟีด | `python .claude/skills/reach/reach.py rss <url>` |
| เว็บอื่นทั้งหมด | `python .claude/skills/reach/reach.py web <url>` |

## กติกา
- ถ้าเจอ ERROR ให้รัน `python .claude/skills/reach/reach.py doctor` แล้วบอกผู้ใช้ว่าขาดอะไร
- ถ้า yt-dlp ไม่มี ให้ลอง `pip install yt-dlp` ก่อนหนึ่งครั้ง
- สรุปเป็นภาษาไทย และแนบลิงก์ต้นทางท้ายคำตอบ
- ห้ามใช้ cookie หรือรหัสผ่านของผู้ใช้
