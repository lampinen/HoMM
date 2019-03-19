import numpy

class polynomial_family(object):
    def __init__(self, num_variables, max_degree=2):
        self.variables = ["X%i" % i for i in range(num_variables)]
        self.max_degree = max_degree

        if (max_degree < 1):
            raise ValueError("Max degree must be at least 1")
        self.terms = [["1"], [(x,) for x in self.variables]]
        for i in range(2, max_degree + 1):
            self.terms += [[tuple([a] + list(b)) for a in self.variables for b in self.terms[-1] if int(b[0][1:]) >= int(a[1:])]]
        self.flat_terms = [term for deg_terms in self.terms for term in deg_terms]

        def get_term_func(term):
            def term_func(coefficient, values):
                if term == "1":
                    return coefficient
                res = coefficient
                for variable in term:
                   res *= values[variable] 
                return res
            return term_func
        self.term_funcs = {term: get_term_func(term) for term in self.flat_terms} 


    def evaluate(self, coefficients, values):
        res = 0.
        for term, coeff in coefficients.items():
            res += self.term_funcs[term](coeff, values)
        return res


    def add(self, poly1, poly2):
        added_coeffs = {}
        for term in self.flat_terms:
            if term in poly1.coefficients:
                if term in poly2.coefficients:
                    added_coeffs[term] = poly1.coefficients[term] + poly2.coefficients[term]
                else:
                    added_coeffs[term] = poly1.coefficients[term]
            elif term in poly2.coefficients:
                added_coeffs[term] = poly2.coefficients[term]
        return polynomial(self, added_coeffs)


    def _mult_combine_terms(self, term1, term2):
        if term1 == "1":
            return term2
        if term2 == "1":
            return term1

        term1 = list(term1)
        term2 = list(term2)
        new_term = []
        while len(term1) > 0 and len(term2) > 0:
            if int(term1[0][1:]) < int(term2[0][1:]):
                new_term.append(term1.pop(0))
            else:
                new_term.append(term2.pop(0))
        new_term += term1
        new_term += term2
        return tuple(new_term)


    def mult(self, poly1, poly2):
        if poly1.my_max_degree + poly2.my_max_degree > self.max_degree:
            raise ValueError("This multiplication would produce too high-degree a result for this family")

        coeffs = {} 
        for term1, coefficient1 in poly1.coefficients.items(): 
            for term2, coefficient2 in poly2.coefficients.items():
                combined_term = self._mult_combine_terms(term1, term2)
                coeffs[combined_term] = coefficient1 * coefficient2

        return polynomial(self, coeffs)

    
    def _term_to_symbols(self, term):
        if term == "1":
            return ""
        term = list(term)
        res = ""
        curr_var = term.pop(0)
        curr_count = 1 
        while len(term) > 0:
            next_var = term.pop(0)
            if next_var == curr_var:
                curr_count += 1
            else:
                if curr_count == 1:
                    res += curr_var
                else:
                    res += "%s^%i" % (curr_var, curr_count)
                curr_var = next_var 
                curr_count = 1
        if curr_count == 1:
            res += curr_var
        else:
            res += "%s^%i" % (curr_var, curr_count)
        return res
                

    
    def poly_to_symbols(self, poly):
        temp_results = [[] for _ in range(self.max_degree + 1)]
        for term, coeff in poly.coefficients.items():
            degree = 0 if term == "1" else len(term)
            temp_results[degree] += ["%.2f%s" % (coeff, 
                                                 self._term_to_symbols(term))] 
        temp_results = [l for l in temp_results if l != []]

        return " + ".join([" + ".join(deg_results) for deg_results in temp_results])


class polynomial(object):
    def __init__(self, family, coefficients):
        self.family = family
        self.coefficients = coefficients
        self.my_max_degree = max([len(term) for term in self.coefficients.keys()])

    def values_to_dict(self, values):
        return {var: values[i] for i, var in enumerate(self.family.variables)}  

    def evaluate(self, values):
        return self.family.evaluate(self.coefficients, self.values_to_dict(values))

    def __add__(self, poly2):
        return self.family.add(self, poly2)

    def __mul__(self, poly2):
        return self.family.mult(self, poly2)

    def to_symbols(self):
        return self.family.poly_to_symbols(self)


if __name__ == "__main__":
    p_fam = polynomial_family(3, 3)

    coeffs = {("X0", "X0"): 1, 
              "1": 1}
    x2c1 = polynomial(p_fam, coeffs)
    print("x^2 + 1")
    print(x2c1.my_max_degree)
    print(x2c1.evaluate([1, 1, 1]))
    print(x2c1.evaluate([1, 2, 1]))
    print(x2c1.evaluate([2, 1, 1]))
    print(x2c1.evaluate([2, 2, 1]))
    print(x2c1.evaluate([2, 2, 2]))

    print("xy^2 + 2z")
    coeffs = {("X0", "X1", "X1"): 1, 
              ("X2",): 2}
    print(coeffs)
    xy2plus2z = polynomial(p_fam, coeffs)
    print(xy2plus2z.my_max_degree)
    print(xy2plus2z.evaluate([1, 1, 1]))
    print(xy2plus2z.evaluate([1, 2, 1]))
    print(xy2plus2z.evaluate([2, 1, 1]))
    print(xy2plus2z.evaluate([2, 2, 1]))
    print(xy2plus2z.evaluate([2, 2, 2]))

    added = x2c1 + xy2plus2z
    print("sum")
    print(added.coefficients)
    print(added.my_max_degree)
    print(added.evaluate([1, 1, 1]))
    print(added.evaluate([2, 2, 2]))

    try:
        multiplied = x2c1 * xy2plus2z
    except ValueError:
        print("ValueError as expected")

    coeffs = {("X1",): 1, 
              "1": 2}
    yplus2 = polynomial(p_fam, coeffs)

    multiplied = x2c1 * yplus2
    print(multiplied.coefficients)
    print(multiplied.my_max_degree)

    print(x2c1.to_symbols())
    print(yplus2.to_symbols())
    print(multiplied.to_symbols())
