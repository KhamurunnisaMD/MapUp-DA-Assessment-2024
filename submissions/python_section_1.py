def reverse_by_n(list, n):
    result = []
    for i in range(0, len(list), n):
        group = list[i:i + n]
        reversed_group = []
        for j in range(len(group)):
            reversed_group.append(group[len(group) - 1 - j])
        result.extend(reversed_group)
    return result
print(reverse_by_n([1, 2, 3, 4, 5, 6, 7, 8], 3))   
print(reverse_by_n([1, 2, 3, 4, 5], 2))           
print(reverse_by_n([10, 20, 30, 40, 50, 60, 70], 4)) 



