import os
import subprocess


def generate_docs():
    os.chdir('docs')
    subprocess.run(['sphinx-apidoc', '-o', 'source', '../annime/annime'])
    subprocess.run(['make', 'html'])
    os.chdir('..')


if __name__ == '__main__':
    generate_docs()
