
"""
    Ackley function (see https://www.sfu.ca/~ssurjano/ackley.html)
    param: [a, b, c]
"""
function ackley_grad_x(x, y, param; epsilon=1e-8)
    a, b, c = param

    sum_square = x^2 + y^2 + epsilon

    term1 = a * b * x * exp(-b * sqrt(sum_square / 2)) * (1 / sqrt(2 * sum_square))
    term2 = (c / 2) * sin(c * x) * exp(0.5 * (cos(c * x) + cos(c * y)))
    
    return term1 + term2
end
function ackley_grad_y(x, y, param; epsilon=1e-8)
    a, b, c = param

    sum_square = x^2 + y^2 + epsilon

    term1 = a * b * y * exp(-b * sqrt(sum_square / 2)) * (1 / sqrt(2 * sum_square))
    term2 = (c / 2) * sin(c * y) * exp(0.5 * (cos(c * x) + cos(c * y)))
    
    return term1 + term2
end

function ackley(x, y, param)
    a, b, c = param

    term1 = -a * exp(-b * sqrt((x^2 + y^2) / 2))
    term2 = -exp(0.5 * (cos(c * x) + cos(c * y)))
    
    return term1 + term2 + a + exp(1)
end

