import numpy

class polynomial_family(object):
    def __init__(self, num_variables, max_degree=2):
        self.variables = ["X%i" % i for i in range(num_variables)]
        self.max_degree = max_degree

        if (max_degree < 1):
            raise ValueError("Max degree must be at least 1")
        self.terms = [["1"], [[x] for x in self.variables]]
        for i in range(2, max_degree + 1):
            self.terms += [[[a] + b for a in self.variables for b in self.terms[-1] if int(b[0][1:]) >= int(a[1:])]]

        def get_term_func(term):
            def term_func(coefficient, values):
                if term == "1":
                    return coefficient
                res = coefficient
                for variable in term:
                   res *= values[variable] 
                return res
            return term_func
        self.term_funcs = [[get_term_func(term) for term in deg_terms] for deg_terms in self.terms]  


    def evaluate(self, coefficients, values):
        res = 0.
        for deg_i in range(self.max_degree + 1):
            for c, term_func in zip(coefficients[deg_i], self.term_funcs[deg_i]):
                res += term_func(c, values)
        return res

    def zero_coefficients(self):
        return [[0 for term in deg_terms] for deg_terms in self.terms] 


class polynomial(object):
    def __init__(self, family, coefficients):
        self.family = family
        self.coefficients = coefficients

    def values_to_dict(self, values):
        return {var: values[i] for i, var in enumerate(self.family.variables)}  

    def evaluate(self, values):
        return self.family.evaluate(self.coefficients, self.values_to_dict(values))


if __name__ == "__main__":
    p_fam = polynomial_family(3, 3)

    coeffs = p_fam.zero_coefficients()
    coeffs[0][0] = 1.
    coeffs[2][0] = 1.
    x2c1 = polynomial(p_fam, coeffs)
    print("x^2 + 1")
    print(x2c1.evaluate([1, 1, 1]))
    print(x2c1.evaluate([1, 2, 1]))
    print(x2c1.evaluate([2, 1, 1]))
    print(x2c1.evaluate([2, 2, 1]))

    print("xy^2 + 2z")
    coeffs = p_fam.zero_coefficients()
    coeffs[1][2] = 2.
    coeffs[3][3] = 1.
    print(coeffs)
    xy2plus2z = polynomial(p_fam, coeffs)
    print(xy2plus2z.evaluate([1, 1, 1]))
    print(xy2plus2z.evaluate([1, 2, 1]))
    print(xy2plus2z.evaluate([2, 1, 1]))
    print(xy2plus2z.evaluate([2, 2, 1]))
