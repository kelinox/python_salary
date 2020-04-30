def predict_salary(exp, weight, bias):
    return weight*exp + bias


def cost_function(experiences, salary, weight, bias):
    nb_lines = len(experiences)
    total_error = 0.0

    # For every line in our dataset we measure the error
    for i in range(nb_lines):
        total_error += (salary[i] - (weight*experiences[i] + bias))**2

    # Then we mesure the average error
    return total_error / nb_lines


def update_weights(experiences, salaries, weight, bias, learning_rate):
    weight_deriv = 0
    bias_deriv = 0
    nb_lines = len(experiences)

    # For every line in our dataset we are calculating by how much we need to
    # increase our bias and weight
    for i in range(nb_lines):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        weight_deriv += 2*experiences[i] * (salaries[i] - (weight*experiences[i] + bias))

        # -2(y - (mx + b))
        bias_deriv += 2*(salaries[i] - (weight*experiences[i] + bias))

    # We add our learning average gradient descent multiply by our learning rate
    weight += (weight_deriv / nb_lines) * learning_rate
    bias += (bias_deriv / nb_lines) * learning_rate

    return weight, bias


def train(experiences, salaries, weight, bias, learning_rate, iters):
    cost_history = []

    # This is the number of iterations we will train our model
    for i in range(iters):
        # We update weight at every iterations
        weight, bias = update_weights(
            experiences,
            salaries,
            weight,
            bias,
            learning_rate)

        # Calculate cost for auditing purposes
        cost = cost_function(experiences, salaries, weight, bias)
        cost_history.append(cost)

        # Log Progress
        if i % 10 == 0:
            print("iterations number: {:d}    weight={:.2f}    bias={:.4f}    cost={:.2}"
                .format(i, weight, bias, cost))

    return weight, bias, cost_history


experiences = [
    0,
    1,
    2,
    3,
    4
]
salaries = [
    30,
    35,
    40,
    45,
    50
]

weight, bias, cost_history = train(experiences, salaries, 1, 1, 0.1, 230)
predicted_salary = predict_salary(5, weight, bias)

print("\n\nPredicted salary for {:d} years of experience: {:.2f}"
    .format(5, predicted_salary))
