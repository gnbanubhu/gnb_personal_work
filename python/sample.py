def find_max(numbers):
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num


numbers = [3, 67, 12, 89, 45, 23, 100, 7, 56]
print(f"Array   : {numbers}")
print(f"Max number: {find_max(numbers)}")
