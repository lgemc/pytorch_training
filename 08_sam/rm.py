import json
import fmm
import os
import shutil


def get_name(cn, ln ,n, code):
    name = 'V' + str(cn) + '_L' + str(ln) + '_R' + str(n) + '_' + str(int(code))
    return name


def save(response, cn, ln, n, code):
    name = get_name(cn, ln, n, code)
    file_name = name + '.json'
    mapped_code = fmm.map_code(str(code))
    mapped_code = mapped_code.item()
    answer = {'ans': response, 
              'ansid': mapped_code}
    shutil.unpack_archive('answers.zip', 'answers', 'zip')
    with open(os.path.join('answers', file_name), 'w') as s:
        json.dump(answer, s)
    s.close()
    shutil.make_archive('answers', 'zip', 'answers')
    shutil.rmtree('answers')