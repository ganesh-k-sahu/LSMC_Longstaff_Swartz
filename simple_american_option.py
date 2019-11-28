import numpy as np
import copy
from sklearn.linear_model import LinearRegression

price_sims = np.array([
    [1.00,	1.09, 1.08,	1.34],
    [1.00, 1.16, 1.26, 1.54],
    [1.00, 1.22, 1.07, 1.03],
    [1.00, 0.93, 0.97, 0.92],
    [1.00, 1.11, 1.56, 1.54],
    [1.00, 0.76, 0.77, 0.9],
    [1.00, 0.92, 0.84, 1.01],
    [1.00, 0.88, 1.22, 1.34]
    ])

row_count, column_count = price_sims.shape
strike = 1.1
risk_less_rate = 0.06

# This variable will be updated with conditional option cash flow after each time step of backward induction method
option_cash_flow_matrix = np.zeros(price_sims.shape)


def put_option_payoff(x):
    payoff = max((strike - x), 0.0)
    return payoff


put_option_payoff_vectorised = np.vectorize(put_option_payoff)


def initialise_option_cash_flow_matrix():
    # This takes the matrix of zeros and replaces the last column with the terminal pay offs
    vector_price_sim_time_step_last = price_sims[:, [column_count-1]]
    pay_off_vector = put_option_payoff_vectorised(vector_price_sim_time_step_last)
    option_cash_flow_matrix[:, [column_count-1]] = pay_off_vector
    return None


def calculate_discount_rate_vector_for_given_time_step(time_step_away_from_terminal):
    a = [np.exp(-risk_less_rate * (index + 1) * time_step_away_from_terminal) for index in range(column_count)]
    a.reverse()
    discount_rate_vector_for_given_time_step = np.array([a])
    return discount_rate_vector_for_given_time_step


# Populate the variables with zero values with an initial data relevant for terminal time step
initialise_option_cash_flow_matrix()


def calc_at_time_step(time_step):
    # Step 1: Extract all price simulations at given time step
    vector_price_sim_t = price_sims[:, [time_step]]

    # Step 2: Calculate the square of the price at given time step. To calculate the conditional expected value.
    vector_price_sim_t_squared = np.power(vector_price_sim_t,2)

    # Step 3: Calculate the exercise pay-off value at time for time step. This data will be used to calculate
    #         regression function for cases where pay-offs are positive
    vector_exercise_value = put_option_payoff_vectorised(vector_price_sim_t)

    # Step 4: Calculate the discounted value of all future pay-offs. (time_step - 1) gives the time steps away from last
    discount_rate_vector_for_time_step = calculate_discount_rate_vector_for_given_time_step(time_step - 1)

    # Step 5: Calculate PV of payoffs future to this time step. Discounted future pay offs
    discounted_option_cash_flow_matrix = discount_rate_vector_for_time_step * option_cash_flow_matrix

    # Step 6: Create a 1D vector which gives the PV of all future pay-off from the future discounted cash flow matrix
    vector_discounted_option_cash_flow_matrix = np.sum(discounted_option_cash_flow_matrix, axis=1)
    vector_discounted_option_cash_flow_matrix = np.array([vector_discounted_option_cash_flow_matrix]).transpose()

    # Step 7: Stack 4 columns of data, so that it can be filtered at later step
    stack_matrix = np.hstack([vector_exercise_value,vector_discounted_option_cash_flow_matrix,
                              vector_price_sim_t, vector_price_sim_t_squared ])

    # Step 8: select rows where payoff at current time step is positive (i.e. first column is non zero)
    non_zero_pay_off_cases_t = stack_matrix[stack_matrix[:, 0] != 0]

    # Step 9: Extract the x and y vectors to perform linear regression. Generate conditional expectation function
    #         Remember that the regression function is calculated on positive payoffs at given time step
    y_col = non_zero_pay_off_cases_t[:, [1]]
    x_col = non_zero_pay_off_cases_t[:, [2, 3]]
    input_set_for_conditional_expectation = np.hstack([vector_price_sim_t, vector_price_sim_t_squared])
    reg = LinearRegression().fit(x_col, y_col)

    # Step 10: Calculate the expected value of continuation using the above regression function of dependent variables
    vector_conditional_expected_value_of_continuation = reg.predict(input_set_for_conditional_expectation)

    # Step 11: Stack two vectors of exercise value and continuation value (conditional expectation)
    #          along with a column with boolean values which check positive exercise value over continued value
    stack_matrix_2 = np.hstack([vector_exercise_value, vector_conditional_expected_value_of_continuation,
                                vector_exercise_value > vector_conditional_expected_value_of_continuation])

    # Step 12: Take all the rows where the first column is 0, and change the Third column of each of those rows to 0:
    #          This is essentially a filter for out-of-money pay-off at current time step. Third column is set to 0.
    stack_matrix_2[stack_matrix_2[:, 0] == 0, 2] = 0

    # Step 13: Take all the rows where the third column is 0, and change the first column of each of those rows to 0:
    stack_matrix_2[stack_matrix_2[:, 2] == 0, 0] = 0

    # Step 14: Replace the column for current time step with the zeroth column of stack_matrix_2
    option_cash_flow_matrix[:, 2] = stack_matrix_2[:, 0]

    # Step 15: For the columns of option cash flow, where there are non-zero values in column of current time step
    #          make entire row equal to zero
    option_cash_flow_matrix[[option_cash_flow_matrix[:, 2] != 0]] = 0

    # Step 16: Final update to the option pay off matrix - Repeat step 14
    #          Replace the column for current time step with the zeroth column of stack_matrix_2
    option_cash_flow_matrix[:, 2] = stack_matrix_2[:, 0]

    return None

print(option_cash_flow_matrix)
print('----------------------------------------')
calc_at_time_step(2)
print(option_cash_flow_matrix)
print('----------------------------------------')
calc_at_time_step(3)
print(option_cash_flow_matrix)
