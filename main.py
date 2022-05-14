import random
from deap import creator, base, tools, algorithms
import matplotlib.pyplot as plt
import sys
import numpy as np

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=24 * 4)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

DEATH_PENALTY = sys.maxsize

water_consumption = [60, 50, 40, 30, 40, 50, 60, 100, 120, 150, 160, 160, 160, 130, 130, 150, 150, 150, 140, 130, 120,
                     100, 80, 70]
hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
pump_water_output = [0, 10, 30, 40, 50, 60, 80, 90, 100, 110, 130, 140, 150, 160, 180, 190]
wather_energy = [0, 12, 30, 42, 44, 56, 74, 86, 80, 92, 111, 124, 127, 141, 165, 182]


def evalOneMax(individual, present=False):
    energy_price = 0
    tank_volume_begin = 300
    tank_volume_current = tank_volume_begin
    tank_volume_min = 0
    tank_volume_max = 800
    tank_sequence_timestamp = []
    tank_volume_timestamp = []
    water_consumption_timestamp = []
    pump_water_output_timestamp = []

    for i in range(0, 96):
        if i % 4 == 0:
            bitArr = [individual[i], individual[i + 1], individual[i + 2], individual[i + 3]]
            # [0, 0, 1, 1] -> 3
            pump_sequence = int(bitArr[0] * pow(2, 0) + bitArr[1] * pow(2, 1) + bitArr[2] * pow(2, 2) + bitArr[3] * pow(2, 3))
            tank_sequence_timestamp.append(pump_sequence)
            hour = int(i / 4)

            tank_volume_current -= water_consumption[hour]
            tank_volume_current += pump_water_output[pump_sequence]
            tank_volume_timestamp.append(tank_volume_current)
            water_consumption_timestamp.append(water_consumption[hour])
            pump_water_output_timestamp.append(pump_water_output[pump_sequence])

            if tank_volume_current < tank_volume_min or tank_volume_current > tank_volume_max:
                return (DEATH_PENALTY),

            if 7 <= hour <= 20:
                energy_price += wather_energy[pump_sequence] * 14
            else:
                energy_price += wather_energy[pump_sequence] * 7

            if present:
                print(f"Hour: {hour}, Sequence: {individual[i]}, {individual[i + 1]}, {individual[i + 2]}, {individual[i + 3]}")

    if tank_volume_current < tank_volume_begin:
        energy_price += abs(tank_volume_begin - tank_volume_current) * 15

    if tank_volume_current > tank_volume_begin:
        energy_price -= abs(tank_volume_begin - tank_volume_current) * 6

    if present:
        plt.figure(figsize=(10, 10))
        plt.plot(hours, tank_volume_timestamp)
        plt.plot(hours, water_consumption_timestamp)
        plt.plot(hours, pump_water_output_timestamp)
        plt.xlabel("Hour")
        plt.ylabel("Consumption")
        plt.title(f"Water Consumption")
        plt.legend(['Tank Volume', 'Water Consumption', 'Water Output'], loc='upper left')
        plt.show()

    return (energy_price),


toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=300)

gen_prices = []


best_pop = []
avg_pop = []
worst_pop = []

NGEN = 100
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.3, mutpb=0.5)
    fits = toolbox.map(toolbox.evaluate, offspring)
    best_price = sys.maxsize
    avg_price = 0
    worst_price = 0
    pop_count = 0

    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
        price = fit[0]
        if price != DEATH_PENALTY:
            gen_prices.append(price)
            if price < best_price:
                best_price = price
            if price > worst_price:
                worst_price = price
            avg_price += price
            pop_count += 1

    avg_price /= pop_count
    best_pop.append(abs(best_price - avg_price))
    avg_pop.append(avg_price)
    worst_pop.append(abs(worst_price - avg_price))

    population = toolbox.select(offspring, k=len(population))
top1 = tools.selBest(population, k=1)
evalOneMax(top1[0], True)

plt.figure(figsize=(10, 10))
plt.plot(hours, water_consumption)
plt.xlabel("Hour")
plt.ylabel("Consumption")
plt.title(f"Water Consumption")
plt.legend()
plt.show()


plt.figure(figsize=(10, 10))
plt.plot(gen_prices)
plt.xlabel("Generation")
plt.ylabel("Price")
plt.title(f"Energy price for for next generations. Best {best_price}")
plt.legend()
plt.show()

# example data
ay = np.array(avg_pop)

# example error bar values that vary with x-position

fig, ax = plt.subplots()

# error bar values w/ different -/+ errors that
# also vary with the x-position

asymmetric_error = [np.array(best_pop), np.array(worst_pop)]

x = np.arange(0, len(avg_pop), 1)
ax.errorbar(x=x, y=ay, yerr=asymmetric_error, fmt='o')
ax.set_title('Best-Avg-Worst pop')
# ax.set_yscale('log')
plt.show()

fig, ax = plt.subplots()
ax.errorbar(x=x, y=ay, yerr=asymmetric_error, fmt='o')
ax.set_title('Best-Avg-Worst pop')
ax.set_yscale('log')
plt.show()

print(f"Last gen: worst:{avg_pop[-1] + worst_pop[-1]}, avg:{avg_pop[-1]}, best:{avg_pop[-1] - best_pop[-1]}")
