import json 
import sys
import subprocess

indexes=['exc1', 'exc2', 'pv', 'sst1', 'sst2', 'vip1', 'vip2']
def compare_jsons(json1, json2):
    """Compare two JSONs and return a list of differences."""
    diff = {}
    unionKeys = set(json1.keys()).union(set(json2.keys()))
    for key in unionKeys:
        if key not in json1:
            diff[key] = [None, json2[key]]
        elif key not in json2:
            diff[key] = [json1[key], None]
        elif json1[key] != json2[key]:
            for i,(e1,e2) in enumerate(zip(json1[key], json2[key])):
                if float(e1) != float(e2):
                    diff[f'{key}.{indexes[i]}'] = [e1,e2]
    return diff

def compare_jsons_parent(json1, json2):
    """Compare two JSONs and return a list of differences."""
    diff = {}
    diff_z = compare_jsons(json1['z'], json2['z'])
    diff_dz = compare_jsons(json1['dz'], json2['dz'])
    for key in diff_z:
        diff['z.'+key] = diff_z[key]
    for key in diff_dz:
        diff['dz.'+key] = diff_dz[key]
    return diff


def main(filename1,filename2):
    line_number=0
    with open(filename1) as f1, open(filename2) as f2:
        while True:
            line_number += 1
            l1 = f1.readline()
            l2 = f2.readline()
            if not l1 or not l2:
                print(f'One of the files is empty at line {line_number}')
                return 
            if l1 != l2:
                # print(f"Line {line_number} is different")
                d1 = json.loads(l1[6:])
                d2 = json.loads(l2[6:])
                if d1['t'] != d2['t']:
                    print(f"t is different: {d1['t']} vs {d2['t']} at line {line_number}")
                    return 
                diff = compare_jsons_parent(d1, d2)
                if diff=={}:
                    continue
                print(f"Diff at line {line_number}:")
                for key in diff:
                    print(f"  {key}: {diff[key]}")

     
    

if __name__ == "__main__":
    # subprocess.run(['python start.py | grep json > a.out'])
    # subprocess.run(['python original.py | grep json > b.out'])
    main('a.out', 'b.out')