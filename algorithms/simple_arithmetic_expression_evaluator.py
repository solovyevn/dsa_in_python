#!/usr/bin/env python
# _*_ encoding: utf-8 _*_

"""Implements simple arithmetic expression evaluator based on Dijkstra's idea."""


def evaluate(expr):
    """Evaluates simple fully parenthesized arithmetic expressions.

    Basic algorithm:
        - Ignore left parentheses.
        - Push operands onto operands stack.
        - Push operators onto operators stack.
        - On encountering a right parenthesis, pop an operator and required number of operands
          from the stacks, performe calculation and push result back onto the operands stack.
        - Once the expression is fully parsed, the resulting value is the only operand on the
          operands stack, pop it and return it.

    Args:
        expr (str): String with expression to evaluate.

    Raises:
        ValueError: In case an `expression` is malformed.

    """
    NUMBERS = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    BINARY_OPS = ('-', '+', '/', '*', '^')
    UNARY_OPS = ('sqrt', )  # Not used this way at the moment, should be improved
    operands = []  # Stack
    operators = []  # Stack
    i = 0
    while i<len(expr):
        char = expr[i]
        if char in ('(', ' '):
            pass
        elif char in BINARY_OPS:
            operators.append(char)
        elif char == 's':
            op = ''
            op_start_i = i
            while i != op_start_i + 4:
                op += expr[i]
                i += 1
                if i == len(expr):
                    break
            if op == 'sqrt':
                operators.append(op)
            else:
                raise ValueError(f'Invalid operation at {op_start_i}.')
        elif char in (')', ):
            if operators:
                operator = operators.pop()
            else:
                break
            if operator in BINARY_OPS:
                operand2 = int(operands.pop())
                operand1 = int(operands.pop())
                if operator == '-':
                    operands.append(operand1 - operand2)
                elif operator == '+':
                    operands.append(operand1 + operand2)
                elif operator == '/':
                    operands.append(operand1 // operand2)
                elif operator == '*':
                    operands.append(operand1 * operand2)
                elif operator == '^':
                    operands.append(pow(operand1, operand2))
            else:
                operand = int(operands.pop())
                if operator == 'sqrt':
                    operands.append(pow(operand, 1/2))
        else:
            number_start_i = i
            if char in NUMBERS:
                number = char
            else:
                number = ''
            while expr[i+1] in NUMBERS:
                i += 1
                char = expr[i]
                number += char
                if i == len(expr):
                    break
            if number == '':
                raise ValueError(f'Invalid char at {number_start_i}.')
            else:
                operands.append(number)
        i += 1
    result = operands.pop()
    if operands:
        # Should't happen
        raise ValueError('Provided expression has unmatched parentheses.')
    return result


if __name__ == "__main__":
    cases = [
        ('(1+((2+3)*(4*5)))', 101),
        ('((1+sqrt(25))/2)', 3),
        ('(sqrt(9))', 3),
        ('(1+435)', 436),
        ('(11*11)', 121),
        ('(11^2)', 121),
        ('(((12-2)*(3+7))+(10/5))', 102)
    ]
    for case in cases:
        print(case)
        res = evaluate(case[0])
        if res == case[1]:
            print(f'OK: {res}')
        else:
            print(f'FAIL: {res}')
    for case in ['(1+(abs(2)))',
                 '((1+str(25))/2)',
                 '(1+(1-1)',
                 '(1+1-1)']:
        print(case)
        try:
            res = evaluate(case)
        except ValueError as ex:
            print(f'OK: {ex}')

