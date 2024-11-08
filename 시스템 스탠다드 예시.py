# import sys
# number1, number2 = map(int, sys.stdin.readline().strip().split())

# print(f'number1 + number2 = {number1 + number2}')
# print(f'number1 - number2 = {number1 - number2}')
# print(f'number1 * number2 = {number1 * number2}')
# print(f'number1 / number2 = {number1 / number2}')


import sys

# 입력값이 제대로 있는지 확인
try:
    number1, number2 = map(int, sys.stdin.readline().strip().split())
    
    # 계산 후 출력
    print(f'number1 + number2 = {number1 + number2}')
    print(f'number1 - number2 = {number1 - number2}')
    print(f'number1 * number2 = {number1 * number2}')
    print(f'number1 / number2 = {number1 / number2}')
except ValueError:
    print("입력값이 잘못되었습니다. 두 개의 정수를 입력해주세요.")
