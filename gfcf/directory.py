from os.path import join

ROOT_PATH = "/Users/kgh1030/Documents/2023_1/GFCF/sojong"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))