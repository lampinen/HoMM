import numpy as np
import re
from itertools import combinations_with_replacement, chain


def weird_powerset(iterable, max_size=None):
    s = list(iterable)
    if max_size is None:
        max_size = len(s)
    return chain.from_iterable(combinations_with_replacement(s, r) for r in range(max_size+1))


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
            raise ValueError("This multiplication would produce too high-degree a result for this family: %s %s" % (poly1.to_symbols(), poly2.to_symbols()))

        coeffs = {} 
        for term1, coefficient1 in poly1.coefficients.items(): 
            for term2, coefficient2 in poly2.coefficients.items():
                combined_term = self._mult_combine_terms(term1, term2)
                if combined_term in coeffs:
                    coeffs[combined_term] += coefficient1 * coefficient2
                else:
                    coeffs[combined_term] = coefficient1 * coefficient2

        return polynomial(self, coeffs)


    def pow(self, poly, power):
        if poly.my_max_degree**power > self.max_degree:
            raise ValueError("This power would produce too high-degree a result for this family")

        if power != int(power):
            raise NotImplementedError("Non-integer powers are not implemented")

        new_poly = polynomial(self, poly.coefficients)
        for i in range(power-1):
            new_poly = new_poly * poly

        return new_poly 


    def permute_vars(self, poly, permutation):
        permuted_vars = [self.variables[i] for i in permutation]
        new_var = dict(zip(self.variables, permuted_vars))
        new_coeffs = {}
        for term, coef in poly.coefficients.items():
            if term == "1": 
                new_term = "1"
            else:
                new_term = [new_var[var] for var in list(term)]
                new_term.sort(key=lambda x: int(x[1:]))
                new_term = tuple(new_term)
            new_coeffs[new_term] = coef

        return polynomial(self, new_coeffs)

    
    def _term_to_symbols(self, term):
        if term == "1":
            return ""
        term = list(term)
        res = " "
        curr_var = term.pop(0)
        curr_count = 1 
        while len(term) > 0:
            next_var = term.pop(0)
            if next_var == curr_var:
                curr_count += 1
            else:
                if curr_count == 1:
                    res += curr_var + " "
                else:
                    res += "%s ^ %i " % (curr_var, curr_count)
                curr_var = next_var 
                curr_count = 1
        if curr_count == 1:
            res += curr_var + " "
        else:
            res += "%s ^ %i " % (curr_var, curr_count)
        return res

    
    def poly_to_symbols(self, poly, strip_spaces=False):
        temp_results = [[] for _ in range(self.max_degree + 1)]
        for term, coeff in poly.coefficients.items():
            degree = 0 if term == "1" else len(term)
            temp_results[degree] += [" %.2f%s " % (coeff, 
                                                 self._term_to_symbols(term))] 
        temp_results = [l for l in temp_results if l != []]
        res = "+".join(["+".join(deg_results) for deg_results in temp_results])[1:]
        if strip_spaces:
            res = res.replace(" ", "")

        return res 


    def poly_to_coeff_vec(self, poly):
        vec = []

        possible_terms = weird_powerset(self.variables)
        next(possible_terms)

        for term in ["1"] + list(possible_terms):
            if len(term) > self.max_degree:
                break
            if term in poly.coefficients:
                vec.append(poly.coefficients[term])
            else:
                vec.append(0.)
        return np.array([vec])


    def sample_polynomial(self, coefficient_mean=0, coefficient_sd=2.5,
                          intercept_probability=0.5, term_probability=0.5):
        num_relevant_variables = np.random.randint(len(self.variables) + 1)
        coefficients = {}
        while coefficients == {} or (num_relevant_variables > 0 and coefficients.keys() == ["1"]):
            if num_relevant_variables == 0 or np.random.rand() < intercept_probability: 
                constant_coefficient = np.random.randn() * coefficient_sd + coefficient_mean
                coefficients["1"] = constant_coefficient

            relevant_variables = [self.variables[i] for i in sorted(np.random.permutation(len(self.variables))[:num_relevant_variables])]
            possible_terms = weird_powerset(relevant_variables)
            next(possible_terms) # get rid of the empty set
            for term in possible_terms:
                if len(term) > self.max_degree:
                    break
                if np.random.rand() > term_probability:
                    continue
                this_coefficient = np.random.randn() * coefficient_sd + coefficient_mean
                coefficients[term] = this_coefficient 

        return polynomial(self, coefficients)


    def sample_point(self, val_range=5):
        num_relevant_variables = np.random.randint(1, len(self.variables) + 1)
        relevant_variable_indices = sorted(np.random.permutation(len(self.variables))[:num_relevant_variables])
        values = [np.random.rand() * 2 * val_range - val_range if i in relevant_variable_indices else 0 for i in range(len(self.variables))]
        return values


class polynomial(object):
    def __init__(self, family, coefficients):
        self.family = family
        self.coefficients = coefficients
        self.my_max_degree = max([len(term) for term in self.coefficients.keys() if isinstance(term, tuple)] + [0])
        self.relevant_variables = [v for v in self.family.variables if any([v in term for term in self.coefficients])]
        if "1" in self.coefficients:
            self.relevant_variables.append("1")

    def values_to_dict(self, values):
        return {var: values[i] for i, var in enumerate(self.family.variables)}  

    def evaluate(self, values):
        return self.family.evaluate(self.coefficients, self.values_to_dict(values))

    def __add__(self, poly2):
        if isinstance(poly2, int) or isinstance(poly2, float): # constant addition 
            new_coeffs = {t: c for t, c in self.coefficients.items()}
            if "1" in new_coeffs:
                new_coeffs["1"] += poly2
            else:
                new_coeffs["1"] = poly2
            return polynomial(self.family, new_coeffs)
        return self.family.add(self, poly2)

    def __mul__(self, poly2):
        if isinstance(poly2, int) or isinstance(poly2, float): # scalar multiplication
            return polynomial(
                self.family, {t: poly2*c for t, c in self.coefficients.items()})
        return self.family.mult(self, poly2)


    def __pow__(self, power):
        return self.family.pow(self, power) 


    def __eq__(self, poly2):
        return isinstance(poly2, polynomial) and self.family == poly2.family and self.coefficients == poly2.coefficients


    def __ne__(self, poly2):
        return not self.__eq__(poly2)


    def permute_vars(self, permutation):
        return self.family.permute_vars(self, permutation)


    def to_symbols(self, strip_spaces=False):
        return self.family.poly_to_symbols(self, strip_spaces=strip_spaces)


    def to_coeff_vec(self):
        return self.family.poly_to_coeff_vec(self)

    def __str__(self):
        return self.to_symbols(strip_spaces=True)


def stringify_polynomial(p):
    """Helper for printing, etc."""
    return str(p) 


number_regex = re.compile('-?[0-9]+\.[0-9][0-9]')


def intify_polynomial(p):
    """Helper for language inputs"""
    symbs = p.to_symbols(strip_spaces=False)
    symbs = re.sub("\+ -", "-", symbs)
    symbs = symbs.split()
    ints = []
    for x in symbs:
        if number_regex.match(x):
            ints += [vocab_to_int[ch] for ch in list(x)]
        else:
            ints.append(vocab_to_int[x])
    return ints


def get_distinct_random_choices(values, num_choices_per, num_sets,
                                replace=False):
    sets = []
    while len(sets) < num_sets:
        candidate_set = set(np.random.choice(values, num_choices_per,
                                             replace=replace))
        if candidate_set not in sets:
            sets.append(candidate_set)

    return [np.random.permutation(list(s)) for s in sets]


def get_meta_pairings(base_train_tasks, base_eval_tasks, meta_class_train_tasks, meta_class_eval_tasks,
                      meta_map_train_tasks, meta_map_eval_tasks):
    """Gets which tasks map to which other tasks under the meta mappings."""
    all_meta_tasks = meta_class_train_tasks + meta_class_eval_tasks + meta_map_train_tasks + meta_map_eval_tasks 
    meta_pairings = {mt: {"train": [], "eval": []} for mt in all_meta_tasks}
    implied_tasks = {"train": [], "eval": []} 
    for mt in all_meta_tasks:
        for curr_base_tasks, train_or_eval in zip([base_train_tasks,
                                                   base_eval_tasks],
                                                  ["train", "eval"]):
            if mt == "square":
                for poly in curr_base_tasks:
                    if poly.my_max_degree**2 > poly.family.max_degree:
                        continue
                    other = poly ** 2
                    implied_tasks[train_or_eval].append(other)
                    meta_pairings[mt][train_or_eval].append(
                        (stringify_polynomial(poly),
                         stringify_polynomial(other)))
            elif mt[:3] == "add":
                c = float(mt[4:])
                for poly in curr_base_tasks:
                    other = poly + c
                    implied_tasks[train_or_eval].append(other)
                    meta_pairings[mt][train_or_eval].append(
                        (stringify_polynomial(poly),
                         stringify_polynomial(other)))
            elif mt[:4] == "mult":
                c = float(mt[5:])
                for poly in curr_base_tasks:
                    other = poly * c
                    implied_tasks[train_or_eval].append(other)
                    meta_pairings[mt][train_or_eval].append(
                        (stringify_polynomial(poly),
                         stringify_polynomial(other)))
            elif mt[:7] == "permute":
                perm = [int(c) for c in mt[8:]]
                for poly in curr_base_tasks:
                    other = poly.permute_vars(perm)
                    implied_tasks[train_or_eval].append(other)
                    meta_pairings[mt][train_or_eval].append(
                        (stringify_polynomial(poly),
                         stringify_polynomial(other)))
            elif mt == "is_constant_polynomial":
                for poly in curr_base_tasks:
                    truth_val = poly.my_max_degree == 0
                    meta_pairings[mt][train_or_eval].append(
                        (stringify_polynomial(poly),
                         1*truth_val))
            elif mt == "is_intercept_nonzero":
                for poly in curr_base_tasks:
                    truth_val = "1" in poly.coefficients and poly.coefficients["1"] != 0.
                    meta_pairings[mt][train_or_eval].append(
                        (stringify_polynomial(poly),
                         1*truth_val))
            elif mt[:3] == "is_":
                var = mt.split("_")[1]
                for poly in curr_base_tasks:
                    truth_val = var in poly.relevant_variables
                    meta_pairings[mt][train_or_eval].append(
                        (stringify_polynomial(poly),
                         1*truth_val))
            elif mt[:6] == "binary":
                operation = mt[7:]
                if operation not in ["sum", "mult"]:
                    raise ValueError("Unknown meta task: %s" % meta_task)
                pairings = get_distinct_random_choices(
                    values=curr_base_tasks, num_choices_per=2,
                    num_sets=config["num_meta_binary_pairs"], replace=False)
                for poly1, poly2 in pairings:
                    if operation == "sum":
                        other = poly1 + poly2
                    elif operation == "mult":
                        if poly1.my_max_degree + poly2.my_max_degree > poly1.family.max_degree:
                            continue
                        other = poly1 * poly2
                    implied_tasks[train_or_eval].append(other)
                    meta_pairings[mt][train_or_eval].append(
                        (stringify_polynomial(poly1),
                         stringify_polynomial(poly2),
                         stringify_polynomial(other)))

            else:
                raise ValueError("Unknown meta task: %s" % meta_task)

    implied_train_tasks = implied_tasks["train"]
    implied_eval_tasks = implied_tasks["eval"]
    return meta_pairings, implied_train_tasks, implied_eval_tasks

if __name__ == "__main__":
    p_fam = polynomial_family(3, 3)

    coeffs = {("X0", "X0"): 1, 
              "1": 1}
    x2c1 = polynomial(p_fam, coeffs)
    print("x^2 + 1")
    print(x2c1.my_max_degree)
    print(x2c1.relevant_variables)
    print(x2c1.evaluate([1, 1, 1]))
    print(x2c1.evaluate([1, 2, 1]))
    print(x2c1.evaluate([2, 1, 1]))
    print(x2c1.evaluate([2, 2, 1]))
    print(x2c1.evaluate([2, 2, 2]))
    print(x2c1.evaluate(np.array([2, 2, 2])))


    print("xy^2 + 2z")
    coeffs = {("X0", "X1", "X1"): 1, 
              ("X2",): 2}
    print(coeffs)
    xy2plus2z = polynomial(p_fam, coeffs)
    print(xy2plus2z.my_max_degree)
    print(xy2plus2z.relevant_variables)
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

    print("Symbols")
    print(x2c1.to_symbols())
    print(yplus2.to_symbols())
    print(multiplied.to_symbols())
    print(multiplied.to_symbols(strip_spaces=True))

    print("Scalar multiplication")
    print((x2c1 * 2).to_symbols())
    print((yplus2 * 3.4).to_symbols())

    print("Scalar addition")
    print((x2c1  +  2).to_symbols())
    print((yplus2 + 3.4).to_symbols())

    print("permuting")
    permuted1 = yplus2.permute_vars([2, 0, 1])
    print(permuted1.to_symbols())
    permuted_multiplied = multiplied.permute_vars([0, 2, 1])
    print(permuted_multiplied.to_symbols())

    print("Powers")
    print((yplus2 ** 1).to_symbols())
    print((yplus2 ** 2).to_symbols())
    print((yplus2 ** 3).to_symbols())

    print("random samples")
    np.random.seed(0)
    for _ in range(5):
        this_rand_poly = p_fam.sample_polynomial()
        print(this_rand_poly.to_symbols())
        print(this_rand_poly.to_coeff_vec())
