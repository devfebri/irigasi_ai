import os

files = ['build_nb01.py','build_nb02.py','build_nb03.py','build_nb04.py','build_nb05.py']
for fn in files:
    with open(fn, encoding='utf-8') as f:
        txt = f.read()
    # hapus baris print terakhir yang berisi emoji
    lines = txt.splitlines()
    new_lines = []
    for line in lines:
        # ganti emoji checkmark di print statement
        if '\u2705' in line:
            line = line.replace('\u2705', '[OK]')
        if '\U0001f4a7' in line:
            line = line.replace('\U0001f4a7', '[DONE]')
        new_lines.append(line)
    with open(fn, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    print(f'Fixed: {fn}')

print('All done')
