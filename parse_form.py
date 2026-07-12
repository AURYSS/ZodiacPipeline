import re
import json
import sys

html_path = r'C:\Users\Aurora\.gemini\antigravity\brain\d9ae68f0-797f-4ce8-891f-d4bfc9b4b35e\.system_generated\steps\173\content.md'
with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()

match = re.search(r'var FB_PUBLIC_LOAD_DATA_ = (\[.*?\]);\s*</script>', html, re.DOTALL)
if match:
    data = json.loads(match.group(1))
    questions = data[1][1]
    for q in questions:
        title = q[1]
        print(title.encode('ascii', 'ignore').decode())
