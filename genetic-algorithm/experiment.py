import math
import random
import matplotlib.pyplot as plt


# 基因型为表现型 gray 码 (Genotype is the gray code of Phenotype)
# 基因取值范围为 0~255 (The range of Gene is 0 to 255)

# 表现参数，编码为 gray 码染色体 (Representation parameter, encode to gray code chromosome)
def encode(param: int, l=8) -> str:
    gray_dec = param ^ (param >> 1)
    gray_bin = bin(gray_dec)[2:]
    return (l - len(gray_bin)) * '0' + gray_bin


# gray 码染色体，解码为表现参数 (Gray code chromosome, decode to representation parameter)
def decode(individual: str, l=8) -> int:
    gray_dec = int(individual, 2)
    org_dec = gray_dec
    for i in range(l-1):
        gray_dec = gray_dec >> 1
        org_dec = org_dec ^ gray_dec
    return org_dec


# 适应度函数 (The fitness function)
def fit(individual: str) -> float:
    param = decode(individual)
    # 随便整一个函数
    out = 2 * math.exp((-math.pow(param-144, 2))/100) + math.exp((-math.pow(param-189, 2))/500)
    return out


def softmax(fitness: list[float]) -> list[float]:
    denominator = sum([math.exp(fitness[i]) for i in range(len(fitness))])
    return [math.exp(fitness[i]) / denominator for i in range(len(fitness))]


# 选择，选择50个染色体 (Selection, select 50 chromosomes)
def select(population, weights, nselect=50):
    return random.choices(population, weights=weights, k=nselect)


# 交叉，随机选择50对染色体交叉，得到100个新染色体。pc为交叉因子
# (Crossover, select 50 pairs of chromosomes randomly to crossover, then get 100 new chromosomes)
def crossover(population, pc=0.3, ncross=50):
    l = len(population[0])
    new_population = []
    for i in range(ncross):
        pair = random.choices(population, k=2)
        cross_template = random.choices([0, 1], weights=[pc, 1-pc], k=8)
        new_chromosome1 = ''.join([pair[0][j] if cross_template[j] else pair[1][j] for j in range(l)])
        new_chromosome2 = ''.join([pair[1][j] if cross_template[j] else pair[0][j] for j in range(l)])
        new_population.extend([new_chromosome1, new_chromosome2])
    return new_population


# 变异，在100个新染色体中随机改变基因 (Mutation, randomly change the gene in 100 new chromosomes)
def mutate(population, pm=0.002):
    n = len(population)
    l = len(population[0])
    mutation = random.choices([True, False], weights=[pm, 1-pm], k=n*l)
    for i, m in enumerate(mutation):
        if (m):
            chromosome_list = list(population[i//l])
            chromosome_list[i%l] = '1' if chromosome_list[i%l] == '0' else '0'
            population[i//l] = ''.join(chromosome_list)
    return population


def main():
    n = 100
    # 随机生成 n 个染色体 (Generate n chromosomes randomly)
    population = [encode(random.randint(0, 255)) for _ in range(n)]
    print(population)

    gens = 0
    for g in range(200):
        # 计算适应度 (Compute the fitness)
        fitness = [fit(population[i]) for i in range(n)]
        weights = softmax(fitness)
        # 选择，根据适应度对染色体进行自然选择 (Selection, natural select chromosomes according to the fitness)
        selected = select(population, weights, nselect=50)
        # 交叉，融合基因得到下一代染色体 (Crossover, fuse genes to get the next generation of chromosomes)
        crossed = crossover(selected, pc=0.3, ncross=50)
        # 变异，以极小概率随机改动下一代的基因
        # (Mutation, randomly change genes of the next generation in a extremely small probability)
        mutated = mutate(crossed)

        # 迭代 (Iterate)
        population = mutated
        gens += 1
        print(f'Generation {g}')

    result = [decode(population[i], l=8) for i in range(n)]
    plt.hist(result)
    plt.show()


if __name__ == '__main__':
    main()
