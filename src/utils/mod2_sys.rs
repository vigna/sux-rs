#![allow(unexpected_cfgs)]
use crate::{bits::bit_vec::BitVec, traits::Word};
use anyhow::{bail, ensure, Result};
use std::cmp::min;
#[cfg(feature = "time_log")]
use std::time::SystemTime;

/// An equation on **F**~2~
#[derive(Clone, Debug)]
pub struct Modulo2Equation<W: Word = usize> {
    /// The bit vector representing the coefficients (one bit for each variable)
    bit_vector: BitVec,
    /// The constant term
    c: W,
    /// The index of the first variable in the equation, if any
    first_var: Option<u32>,
}

/// Solver for linear systems on **F**~2~
/// Variables are k-dimensional vectors on **F**~2~, with 0 $$\le$$ k $$\le$$ 64
#[derive(Clone, Debug)]
pub struct Modulo2System<W: Word> {
    /// The number of variables
    num_vars: usize,
    /// The equations in the system
    equations: Vec<Modulo2Equation<W>>,
}

impl<W: Word> Modulo2Equation<W> {
    /// Creates a new `Modulo2Equation`.
    ///
    /// # Arguments
    ///
    /// * `c` - The constant term of the equation.
    ///
    /// * `num_vars` - The total number of variables in the equation.
    pub fn new(c: W, num_vars: usize) -> Self {
        Modulo2Equation {
            bit_vector: BitVec::new(num_vars),
            c,
            first_var: None,
        }
    }

    /// Adds a variable to the equation.
    ///
    /// # Arguments
    ///
    /// * `variable` - The index of the variable to be added.
    ///
    /// # Panics
    ///
    /// The method panics if the variable is already present in the equation.
    pub fn add(&mut self, variable: usize) -> &mut Self {
        assert!(
            !self.bit_vector.get(variable),
            "Variable {variable} already in equation"
        );
        self.bit_vector.set(variable, true);
        let variable = variable as u32;
        self.first_var = Some(min(self.first_var.unwrap_or(variable), variable));
        self
    }

    /// Adds another equation to this equation.
    ///
    /// # Arguments
    ///
    /// * `equation` - The equation to be added.
    pub fn add_equation(&mut self, equation: &Modulo2Equation<W>) {
        self.c ^= equation.c;
        let x = self.bit_vector.as_mut();
        let y = equation.bit_vector.as_ref();
        self.first_var = None;
        for i in 0..x.len() {
            x[i] ^= y[i];
        }
        for (i, &w) in x.iter().enumerate() {
            if w != 0 {
                self.first_var = Some(i as u32 * usize::BITS + w.trailing_zeros());
                break;
            }
        }
    }

    /// Checks if the equation is unsolvable.
    fn is_unsolvable(&self) -> bool {
        self.first_var.is_none() && self.c != W::ZERO
    }

    /// Checks if the equation is an identity.
    fn is_identity(&self) -> bool {
        self.first_var.is_none() && self.c == W::ZERO
    }

    /// Returns the modulo-2 scalar product of the two provided bit vectors.
    ///
    /// # Arguments
    ///
    /// * `bits` - A bit vector represented as a slice of `usize`.
    ///
    /// * `values` - A slice of `usize` representing the values associated
    ///   with each variable.
    fn scalar_product(bits: &[usize], values: &[W]) -> W {
        let mut sum = W::ZERO;

        for (i, &word) in bits.iter().enumerate() {
            let offset = i * usize::BITS as usize;
            let mut word = word;
            while word != 0 {
                let lsb = word.trailing_zeros();
                sum ^= values[offset + lsb as usize];
                word &= word - 1;
            }
        }
        sum
    }

    /// Returns a vector of variables present in the equation.
    pub fn variables(&self) -> Vec<usize> {
        (0..self.bit_vector.len())
            .filter(|&x| self.bit_vector.get(x))
            .collect::<Vec<_>>()
    }
}

impl<W: Word> Modulo2System<W> {
    /// Creates a new `Modulo2System`.
    ///
    /// # Arguments
    ///
    /// * `num_vars` - The total number of variables in the system.
    pub fn new(num_vars: usize) -> Self {
        Modulo2System {
            num_vars,
            equations: Vec::new(),
        }
    }

    /// Creates a new `Modulo2System` from existing equations.
    ///
    /// # Arguments
    ///
    /// * `num_vars` - The total number of variables in the system.
    ///
    /// * `equations` - A vector of existing equations.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the number of variables in each equation matches
    /// the number of variables in the system.
    pub unsafe fn from_parts(num_vars: usize, equations: Vec<Modulo2Equation<W>>) -> Self {
        Modulo2System {
            num_vars,
            equations,
        }
    }

    /// Adds an equation to the system.
    pub fn add(&mut self, equation: Modulo2Equation<W>) {
        assert_eq!(equation.bit_vector.len(), self.num_vars, "The number of variables in the equation ({}) does not match the number of variables in the system ({})", equation.bit_vector.len(), self.num_vars);
        self.equations.push(equation);
    }

    /// Checks if a given solution satisfies the system of equations.
    ///
    /// # Arguments
    ///
    /// * `solution` - A slice of `usize` representing the proposed solution.
    pub fn check(&self, solution: &[W]) -> bool {
        assert_eq!(solution.len(), self.num_vars, "The number of variables in the solution ({}) does not match the number of variables in the system ({})", solution.len(), self.num_vars);
        self.equations.iter().all(|eq| {
            eq.c == Modulo2Equation::<W>::scalar_product(eq.bit_vector.as_ref(), solution)
        })
    }

    /// Transforms the system into echelon form.
    fn echelon_form(&mut self) -> Result<()> {
        if self.equations.is_empty() {
            return Ok(());
        }
        'main: for i in 0..self.equations.len() - 1 {
            ensure!(self.equations[i].first_var.is_some());
            for j in i + 1..self.equations.len() {
                // SAFETY: to add the two equations, multiple references to the vector
                // of equations are needed, one of which is mutable
                let eq_j = unsafe { &*(&self.equations[j] as *const Modulo2Equation<W>) };
                let eq_i = &mut self.equations[i];

                let Some(first_var_j) = eq_j.first_var else {
                    panic!("First variable of equation {} is None", j);
                };

                if eq_i.first_var.expect("First var is None") == first_var_j {
                    eq_i.add_equation(eq_j);
                    if eq_i.is_unsolvable() {
                        bail!("System is unsolvable");
                    }
                    if eq_i.is_identity() {
                        continue 'main;
                    }
                }

                if eq_i.first_var.expect("First var is None") > first_var_j {
                    self.equations.swap(i, j)
                }
            }
        }
        Ok(())
    }

    /// Solves the system using Gaussian elimination.
    pub fn gaussian_elimination(&mut self) -> Result<Vec<W>> {
        let mut solution = vec![W::ZERO; self.num_vars];

        self.echelon_form()?;

        self.equations
            .iter()
            .rev()
            .filter(|eq| !eq.is_identity())
            .for_each(|eq| {
                solution[eq.first_var.expect("First variable is None") as usize] =
                    eq.c ^ Modulo2Equation::scalar_product(eq.bit_vector.as_ref(), &solution);
            });
        Ok(solution)
    }

    /// Solves a system using lazy Gaussian elimination.
    ///
    /// # Arguments
    ///
    /// * `system_op` - The system to be solved, if already exists.
    ///
    /// * `var2_eq` - A vector of vectors describing, for each variable, the equations
    ///   in which it appears.
    ///
    /// * `c` - The vector of known terms, one for each equation.
    ///
    /// * `variable` - the variables with respect to which the system should be solved
    pub fn lazy_gaussian_elimination(
        system_op: Option<&mut Modulo2System<W>>,
        mut var_to_eqs: Vec<Vec<usize>>,
        c: Vec<W>,
        variables: Vec<usize>,
    ) -> Result<Vec<W>> {
        let num_equations = c.len();
        let num_vars = var_to_eqs.len();
        if num_equations == 0 {
            return Ok(vec![W::ZERO; num_vars]);
        }

        let mut new_system = Modulo2System::<W>::new(num_vars);
        let build_system = system_op.is_none();
        let system;
        if build_system {
            system = &mut new_system;
            c.iter()
                .for_each(|&x| system.add(Modulo2Equation::new(x, num_vars)));
        } else {
            system = system_op.unwrap()
        }

        #[cfg(feature = "time_log")]
        let mut measures = Vec::new();
        #[cfg(feature = "time_log")]
        {
            measures.push(num_equations as u128);
            measures.push(system.equations[0].variables().len() as u128); //Number of variables per equation
        }
        #[cfg(feature = "time_log")]
        let mut start = SystemTime::now();

        let mut weight: Vec<usize> = vec![0; num_vars];
        let mut priority: Vec<usize> = vec![0; num_equations];

        for &v in variables.iter() {
            let eq = &mut var_to_eqs[v];
            if eq.is_empty() {
                continue;
            }

            let mut curr_eq = eq[0];
            let mut curr_coeff = true;
            let mut j = 0;

            for i in 1..eq.len() {
                if eq[i] != curr_eq {
                    assert!(
                        eq[i] > curr_eq,
                        "Equations indices do not appear in nondecreasing order"
                    );
                    if curr_coeff {
                        if build_system {
                            system.equations[curr_eq].add(v);
                        }
                        weight[v] += 1;
                        priority[curr_eq] += 1;
                        eq[j] = curr_eq;
                        j += 1;
                    }
                    curr_eq = eq[i];
                    curr_coeff = true;
                } else {
                    curr_coeff = !curr_coeff
                }
            }

            if curr_coeff {
                if build_system {
                    system.equations[curr_eq].add(v);
                }
                weight[v] += 1;
                priority[curr_eq] += 1;
                eq[j] = curr_eq;
                j += 1;
            }
            eq.truncate(j);
        }

        let mut variables = vec![0; num_vars];
        {
            let mut count = vec![0; num_equations + 1];

            for x in 0..num_vars {
                count[weight[x]] += 1
            }
            for i in 1..count.len() {
                count[i] += count[i - 1]
            }
            for i in (0..num_vars).rev() {
                count[weight[i]] -= 1;
                variables[count[weight[i]]] = i;
            }
        }

        let mut equation_list: Vec<usize> = (0..priority.len())
            .rev()
            .filter(|&x| priority[x] <= 1)
            .collect();

        let mut dense: Vec<Modulo2Equation<W>> = Vec::new();
        let mut solved: Vec<usize> = Vec::new();
        let mut pivots: Vec<usize> = Vec::new();

        let equations = &mut system.equations;
        let mut idle_normalized = vec![usize::MAX; num_vars.div_ceil(usize::BITS as usize)];

        let mut remaining = equations.len();
        while remaining != 0 {
            if equation_list.is_empty() {
                let mut var = variables.pop().unwrap();
                while weight[var] == 0 {
                    var = variables.pop().unwrap()
                }
                idle_normalized[var / usize::BITS as usize] ^= 1 << (var % usize::BITS as usize);
                var_to_eqs[var].iter().for_each(|&eq| {
                    priority[eq] -= 1;
                    if priority[eq] == 1 {
                        equation_list.push(eq)
                    }
                });
            } else {
                remaining -= 1;
                let first = equation_list.pop().unwrap();

                if priority[first] == 0 {
                    let equation = &mut equations[first];
                    if equation.is_unsolvable() {
                        bail!("System is unsolvable")
                    }
                    if equation.is_identity() {
                        continue;
                    }
                    dense.push(equation.clone());
                } else if priority[first] == 1 {
                    // SAFETY: to add the equations, multiple references to the vector
                    // of equations are needed, one of which is mutable
                    let equation = unsafe { &*(&equations[first] as *const Modulo2Equation<W>) };
                    let mut word_index = 0;
                    while (equation.bit_vector.as_ref()[word_index] & idle_normalized[word_index])
                        == 0
                    {
                        word_index += 1
                    }
                    let pivot = word_index * usize::BITS as usize
                        + (equation.bit_vector.as_ref()[word_index] & idle_normalized[word_index])
                            .trailing_zeros() as usize;
                    pivots.push(pivot);
                    solved.push(first);
                    weight[pivot] = 0;
                    var_to_eqs[pivot]
                        .iter()
                        .filter(|&&eq_idx| eq_idx != first)
                        .for_each(|&eq| {
                            priority[eq] -= 1;
                            if priority[eq] == 1 {
                                equation_list.push(eq)
                            }
                            equations[eq].add_equation(equation);
                        });
                }
            }
        }

        #[cfg(feature = "time_log")]
        {
            measures.push(start.elapsed()?.as_nanos());
            start = SystemTime::now();
        }

        // SAFETY: the usage of the method is safe, as the equations have the right number of variables
        let mut dense_system = unsafe { Modulo2System::from_parts(num_vars, dense) };
        let mut solution = dense_system.gaussian_elimination()?;

        #[cfg(feature = "time_log")]
        {
            measures.push(start.elapsed()?.as_nanos());
            start = SystemTime::now();
        }

        for i in 0..solved.len() {
            let eq = &equations[solved[i]];
            let pivot = pivots[i];
            assert!(solution[pivot] == W::ZERO);
            solution[pivot] =
                eq.c ^ Modulo2Equation::scalar_product(eq.bit_vector.as_ref(), &solution);
        }

        #[cfg(feature = "time_log")]
        {
            measures.push(start.elapsed()?.as_nanos());
            measures.push(measures[2] + measures[3] + measures[4]);

            let mut measures_csv = measures
                .iter()
                .map(|&measure| measure.to_string())
                .collect::<Vec<String>>();
            measures_csv.push(
                (dense_system.equations.len() as f64 / system.equations.len() as f64).to_string(),
            );
            println!("{}", measures_csv.join(","));
        }

        Ok(solution)
    }

    pub fn lazy_gaussian_elimination_constructor(&mut self) -> Result<Vec<W>> {
        let num_vars = self.num_vars;
        let mut var2_eq = vec![Vec::new(); num_vars];
        let mut d = vec![0; num_vars];
        self.equations.iter().for_each(|eq| {
            (0..eq.bit_vector.len())
                .filter(|&x| eq.bit_vector.get(x))
                .for_each(|x| d[x] += 1)
        });

        var2_eq
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| v.reserve_exact(d[i]));

        let mut c = vec![W::ZERO; self.equations.len()];
        self.equations.iter().enumerate().for_each(|(i, eq)| {
            c[i] = eq.c;
            (0..eq.bit_vector.len())
                .filter(|&x| eq.bit_vector.get(x))
                .for_each(|x| var2_eq[x].push(i));
        });
        Modulo2System::<W>::lazy_gaussian_elimination(
            Some(self),
            var2_eq,
            c,
            (0..num_vars).collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equation_builder() {
        let mut eq = Modulo2Equation::<usize>::new(2, 3);
        eq.add(2).add(0).add(1);
        assert_eq!(eq.variables().len(), 3);
        assert_eq!(eq.variables(), vec![0, 1, 2]);
    }

    #[test]
    fn test_equation_addition() {
        let mut eq0 = Modulo2Equation::<usize>::new(2, 11);
        eq0.add(1).add(4).add(9);
        let mut eq1 = Modulo2Equation::new(1, 11);
        eq1.add(1).add(4).add(10);
        eq0.add_equation(&eq1);
        assert_eq!(eq0.variables(), vec![9, 10]);
        assert_eq!(eq0.c, 3);
    }

    #[test]
    fn test_system_one_equation() {
        let mut system = Modulo2System::<usize>::new(2);
        let mut eq = Modulo2Equation::new(2, 2);
        eq.add(0);
        system.add(eq);
        let solution = system.lazy_gaussian_elimination_constructor();
        assert!(solution.is_ok());
        assert!(system.check(&solution.unwrap()));
    }

    #[test]
    fn test_impossible_system() {
        let mut system = Modulo2System::<usize>::new(1);
        let mut eq = Modulo2Equation::new(2, 1);
        eq.add(0);
        system.add(eq);
        eq = Modulo2Equation::new(1, 1);
        eq.add(0);
        system.add(eq);
        let solution = system.lazy_gaussian_elimination_constructor();
        assert!(solution.is_err());
    }

    #[test]
    fn test_redundant_system() {
        let mut system = Modulo2System::<usize>::new(1);
        let mut eq = Modulo2Equation::new(2, 1);
        eq.add(0);
        system.add(eq.clone());
        system.add(eq);
        let solution = system.lazy_gaussian_elimination_constructor();
        assert!(solution.is_ok());
        assert!(system.check(&solution.unwrap()));
    }

    #[test]
    fn test_small_system() {
        let mut system = Modulo2System::<usize>::new(11);
        let mut eq = Modulo2Equation::new(0, 11);
        eq.add(1).add(4).add(10);
        system.add(eq);
        eq = Modulo2Equation::new(2, 11);
        eq.add(1).add(4).add(9);
        system.add(eq);
        eq = Modulo2Equation::new(0, 11);
        eq.add(0).add(6).add(8);
        system.add(eq);
        eq = Modulo2Equation::new(1, 11);
        eq.add(0).add(6).add(9);
        system.add(eq);
        eq = Modulo2Equation::new(2, 11);
        eq.add(2).add(4).add(8);
        system.add(eq);
        eq = Modulo2Equation::new(0, 11);
        eq.add(2).add(6).add(10);
        system.add(eq);

        let solution = system.lazy_gaussian_elimination_constructor();
        assert!(solution.is_ok());
        assert!(system.check(&solution.unwrap()));
    }
}
