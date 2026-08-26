import random

def simulate_recovery(cart_amount, recovery_probability, action):
    if action == "DO_NOT_INTERVENE":
        return 0

    recovered = random.random() < recovery_probability

    if recovered:
        return cart_amount

    return 0