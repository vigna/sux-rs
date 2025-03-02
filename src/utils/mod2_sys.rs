/*
 *
 * SPDX-FileCopyrightText: 2025 Dario Moschetti
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(unexpected_cfgs)]
use crate::{bits::bit_vec::BitVec, traits::Word};
use anyhow::{bail, ensure, Result};
use arbitrary_chunks::ArbitraryChunks;
use core::panic;
use std::cmp::min;
#[cfg(feature = "time_log")]
use std::time::SystemTime;

/// An equation on **F**~2~
#[derive(Clone, Debug)]
pub struct Modulo2Equation<W: Word = usize, B = Vec<usize>> {
    /// The bit vector representing the coefficients (one bit for each variable)
    bit_vector: BitVec<B>,
    /// The constant term
    c: W,
    /// The index of the first variable in the equation, if any
    first_var: Option<u32>,
}

/// Solver for linear systems on **F**~2~
/// Variables are k-dimensional vectors on **F**~2~, with 0 $$\le$$ k $$\le$$ 64
#[derive(Clone, Debug)]
pub struct Modulo2System<W: Word = usize, B: AsRef<[usize]> + AsMut<[usize]> = Vec<usize>> {
    /// The number of variables
    num_vars: usize,
    /// The equations in the system
    equations: Vec<Modulo2Equation<W, B>>,
}

impl<W: Word> Modulo2Equation<W, Vec<usize>> {
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
}

impl<W: Word, B: AsRef<[usize]> + AsMut<[usize]>> Modulo2Equation<W, B> {
    pub fn to_owned(&self) -> Modulo2Equation<W, Vec<usize>> {
        Modulo2Equation {
            bit_vector: self.bit_vector.to_owned(),
            c: self.c,
            first_var: self.first_var,
        }
    }

    /// Creates a new `Modulo2Equation` from its components.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check the validity of the provided components.
    ///
    /// # Arguments
    ///
    /// * `bit_vector` - The bit vector representing the variables in the equation.
    /// * `c` - The constant term of the equation.
    /// * `first_var` - The index of the first variable in the equation, if any.
    pub unsafe fn from_parts(bit_vector: BitVec<B>, c: W, first_var: Option<u32>) -> Self {
        Modulo2Equation {
            bit_vector,
            c,
            first_var,
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
    pub fn add_equation(&mut self, equation: &Modulo2Equation<W, B>) {
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

impl<W: Word, B: AsRef<[usize]> + AsMut<[usize]>> Modulo2System<W, B> {
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
    pub unsafe fn from_parts(num_vars: usize, equations: Vec<Modulo2Equation<W, B>>) -> Self {
        Modulo2System {
            num_vars,
            equations,
        }
    }

    /// Adds an equation to the system.
    pub fn add(&mut self, equation: Modulo2Equation<W, B>) {
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
            eq.c == Modulo2Equation::<W, B>::scalar_product(eq.bit_vector.as_ref(), solution)
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
                let eq_j = unsafe { &*(&self.equations[j] as *const Modulo2Equation<W, B>) };
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
                solution[eq.first_var.expect("First variable is None") as usize] = eq.c
                    ^ Modulo2Equation::<W, B>::scalar_product(eq.bit_vector.as_ref(), &solution);
            });
        Ok(solution)
    }

    /// Solves a system using lazy Gaussian elimination.
    ///
    /// # Arguments
    ///
    /// * `var2_eq` - A vector of vectors describing, for each variable, the equations
    ///   in which it appears.
    ///
    /// * `c` - The vector of known terms, one for each equation.
    pub fn lazy_gaussian_elimination<A: AsRef<[usize]>, V: AsRef<[A]>>(
        var_to_eqs: V,
        c: Vec<W>,
    ) -> Result<Vec<W>> {
        let num_equations = c.len();
        let var_to_eqs = var_to_eqs.as_ref();
        let num_vars = var_to_eqs.len();
        if num_equations == 0 {
            return Ok(vec![W::ZERO; num_vars]);
        }

        let mut system = Modulo2System::<W, &mut [usize]>::new(num_vars);
        let size_per_eq = num_vars.div_ceil(usize::BITS as usize);
        let slice_size = size_per_eq * num_equations;
        let mut equations_bits = vec![0usize; slice_size];

        equations_bits
            .chunks_mut(size_per_eq)
            .zip(c.iter())
            // SAFETY: num_vars is at most the number of bits in chunk
            .for_each(|(chunk, &c)| unsafe {
                let bv = BitVec::from_raw_parts(chunk, num_vars);
                system.add(Modulo2Equation::<W, &mut [usize]>::from_parts(bv, c, None));
            });

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

        for v in 0..num_vars {
            let eq = &var_to_eqs[v].as_ref();
            if eq.is_empty() {
                continue;
            }

            system.equations[eq[0]].add(v);
            weight[v] += 1;
            priority[eq[0]] += 1;

            for i in 1..eq.len() {
                if eq[i] != eq[i - 1] {
                    assert!(
                        eq[i] > eq[i - 1],
                        "Equations indices do not appear in nondecreasing order"
                    );
                    system.equations[eq[i]].add(v);
                    weight[v] += 1;
                    priority[eq[i]] += 1;
                } else {
                    panic!("Equation {} appears more than once in the list of equations for variable {}", eq[i], v);
                }
            }
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
                var_to_eqs[var].as_ref().iter().for_each(|&eq| {
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
                    dense.push(equation.to_owned());
                } else if priority[first] == 1 {
                    // SAFETY: to add the equations, multiple references to the vector
                    // of equations are needed, one of which is mutable
                    let equation =
                        unsafe { &*(&equations[first] as *const Modulo2Equation<W, &mut [usize]>) };
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
                        .as_ref()
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
                eq.c ^ Modulo2Equation::<W, B>::scalar_product(eq.bit_vector.as_ref(), &solution);
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
        Modulo2System::<W>::lazy_gaussian_elimination(var2_eq, c)
    }
}

pub fn build_var_to_eqs<'a, W: Word, I: Iterator<Item = (impl IntoIterator<Item = usize>, W)>>(
    num_vars: usize,
    get_iter: impl Fn() -> I,
    backend: &'a mut Vec<usize>,
    const_terms: &mut Vec<W>,
) -> Vec<&'a mut [usize]> {
    let mut var_count = vec![0usize; num_vars];
    let mut effective_variables = 0;
    for (i, (it, c)) in get_iter().enumerate() {
        const_terms[i] = c;
        for var in it.into_iter() {
            var_count[var] += 1;
            effective_variables += 1;
        }
    }

    backend.resize(effective_variables, 0);
    let mut var_to_eq: Vec<&mut [usize]> = Vec::with_capacity(num_vars);

    backend.arbitrary_chunks_mut(&var_count).for_each(|chunk| {
        var_to_eq.push(chunk);
    });

    let mut var_indices = vec![0usize; num_vars];
    for (i, (it, _)) in get_iter().enumerate() {
        for var in it.into_iter() {
            var_to_eq[var][var_indices[var]] = i;
            var_indices[var] += 1;
        }
    }
    var_to_eq
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

    #[test]
    fn test_var_to_vec_builder() {
        let iterator = vec![
            (vec![1usize, 4, 10], 0),
            (vec![1, 4, 9], 1),
            (vec![0, 6, 8], 2),
            (vec![0, 6, 9], 3),
            (vec![2, 4, 8], 4),
            (vec![2, 6, 10], 5),
        ];
        let mut bitvec: Vec<usize> = vec![];
        let mut c = vec![0_usize; 6];
        let var_to_eqs = build_var_to_eqs(11, || iterator.clone().into_iter(), &mut bitvec, &mut c);
        let expected_res = vec![
            vec![2, 3],
            vec![0, 1],
            vec![4, 5],
            vec![],
            vec![0, 1, 4],
            vec![],
            vec![2, 3, 5],
            vec![],
            vec![2, 4],
            vec![1, 3],
            vec![0, 5],
        ];
        var_to_eqs
            .iter()
            .zip(expected_res.iter())
            .for_each(|(v, e)| v.iter().zip(e.iter()).for_each(|(x, y)| assert_eq!(x, y)));
        assert_eq!(c, vec![0, 1, 2, 3, 4, 5]);
    }

    // Helper struct that implements the needed trait bounds
    struct IndexIterator(Vec<usize>);

    // Implement IntoIterator for &mut IndexIterator that produces usize values
    impl<'a> IntoIterator for &'a mut IndexIterator {
        type Item = usize;
        type IntoIter = std::iter::Copied<std::slice::Iter<'a, usize>>;

        fn into_iter(self) -> Self::IntoIter {
            self.0.iter().copied()
        }
    }
}
