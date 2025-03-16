/*
 *
 * SPDX-FileCopyrightText: 2025 Dario Moschetti
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(unexpected_cfgs)]
#![allow(clippy::comparison_chain)]
use std::ptr;

use crate::{bit_vec, traits::Word};
use anyhow::{bail, ensure, Result};
use arbitrary_chunks::ArbitraryChunks;

/// An equation on `W::BITS`-dimensional vectors with coefficients in **F**₂.
#[derive(Clone, Debug)]
pub struct Modulo2Equation<W: Word = usize> {
    /// The variables in increasing order.
    vars: Vec<u32>,
    /// The constant term
    c: W,
}

/// A system of [equations](struct.Modulo2Equation.html).
#[derive(Clone, Debug)]
pub struct Modulo2System<W: Word = usize> {
    /// The number of variables.
    num_vars: usize,
    /// The equations in the system.
    equations: Vec<Modulo2Equation<W>>,
}

impl<W: Word> Modulo2Equation<W> {
    /// Creates a new equation with given variables constant term.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the variables are sorted.
    pub unsafe fn from_parts(vars: Vec<u32>, c: W) -> Self {
        debug_assert!(vars.iter().is_sorted());
        Modulo2Equation { vars, c }
    }

    #[inline(always)]
    unsafe fn add_ptr(
        mut left: *const u32,
        left_end: *const u32,
        mut right: *const u32,
        right_end: *const u32,
        mut dst: *mut u32,
    ) -> *mut u32 {
        while left != left_end && right != right_end {
            let less = *left <= *right;
            let more = *left >= *right;

            let src = if less { left } else { right };
            *dst = *src;

            left = left.add(less as usize);
            right = right.add(more as usize);
            dst = dst.add((less ^ more) as usize);
        }

        let rem_left = left_end.offset_from(left) as usize;
        ptr::copy_nonoverlapping(left, dst, rem_left);
        dst = dst.add(rem_left);
        let rem_right = right_end.offset_from(right) as usize;
        ptr::copy_nonoverlapping(right, dst, rem_right);
        dst = dst.add(rem_right);
        dst
    }

    pub fn add(&mut self, other: &Modulo2Equation<W>) {
        let left_range = self.vars.as_ptr_range();
        let left = left_range.start;
        let left_end = left_range.end;
        let right_range = other.vars.as_ptr_range();
        let right = right_range.start;
        let right_end = right_range.end;
        let mut vars = Vec::with_capacity(self.vars.len() + other.vars.len());
        let dst = vars.as_mut_ptr();

        unsafe {
            let copied =
                Self::add_ptr(left, left_end, right, right_end, dst).offset_from(dst) as usize;
            vars.set_len(copied);
        }

        self.vars = vars;
        self.c ^= other.c;
    }

    /// Checks whether the equation is unsolvable.
    fn is_unsolvable(&self) -> bool {
        self.vars.is_empty() && self.c != W::ZERO
    }

    /// Checks whether the equation is an identity.
    fn is_identity(&self) -> bool {
        self.vars.is_empty() && self.c == W::ZERO
    }

    /// Evaluates the XOR of the values associated to the given variables.
    fn eval_vars(vars: impl AsRef<[u32]>, values: impl AsRef<[W]>) -> W {
        let mut sum = W::ZERO;
        for &var in vars.as_ref() {
            sum ^= values.as_ref()[var as usize];
        }
        sum
    }
}

impl<W: Word> Modulo2System<W> {
    pub fn new(num_vars: usize) -> Self {
        Modulo2System {
            num_vars,
            equations: vec![],
        }
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn num_equations(&self) -> usize {
        self.equations.len()
    }

    /// Creates a new `Modulo2System` from existing equations.
    ///
    /// # Safety
    ///
    /// The caller must ensure that variables in each equation match the number
    /// of variables in the system.
    pub unsafe fn from_parts(num_vars: usize, equations: Vec<Modulo2Equation<W>>) -> Self {
        Modulo2System {
            num_vars,
            equations,
        }
    }

    /// Adds an equation to the system.
    pub fn push(&mut self, equation: Modulo2Equation<W>) {
        self.equations.push(equation);
    }

    /// Checks if a given solution satisfies the system of equations.
    pub fn check(&self, solution: &[W]) -> bool {
        assert_eq!(solution.len(), self.num_vars, "The number of variables in the solution ({}) does not match the number of variables in the system ({})", solution.len(), self.num_vars);
        self.equations
            .iter()
            .all(|eq| eq.c == Modulo2Equation::<W>::eval_vars(&eq.vars, solution))
    }

    /// Puts the system into echelon form.
    fn echelon_form(&mut self) -> Result<()> {
        let equations = &mut self.equations;
        if equations.is_empty() {
            return Ok(());
        }
        'main: for i in 0..equations.len() - 1 {
            ensure!(!equations[i].vars.is_empty());
            for j in i + 1..equations.len() {
                // SAFETY: to add the two equations, multiple references to the vector
                // of equations are needed, one of which is mutable
                let eq_j = unsafe { &*(&equations[j] as *const Modulo2Equation<W>) };
                let eq_i = &mut equations[i];

                let first_var_j = eq_j.vars[0];

                if eq_i.vars[0] == first_var_j {
                    eq_i.add(eq_j);
                    if eq_i.is_unsolvable() {
                        bail!("System is unsolvable");
                    }
                    if eq_i.is_identity() {
                        continue 'main;
                    }
                }

                if eq_i.vars[0] > first_var_j {
                    equations.swap(i, j)
                }
            }
        }
        Ok(())
    }

    /// Solves the system using Gaussian elimination.
    pub fn gaussian_elimination(&mut self) -> Result<Vec<W>> {
        self.echelon_form()?;
        let mut solution = vec![W::ZERO; self.num_vars];
        self.equations
            .iter()
            .rev()
            .filter(|eq| !eq.is_identity())
            .for_each(|eq| {
                solution[eq.vars[0] as usize] =
                    eq.c ^ Modulo2Equation::<W>::eval_vars(&eq.vars, &solution);
            });
        Ok(solution)
    }

    /// Builds the data structures needed for lazy Gaussian elimination.
    ///
    /// This method returns the variable-to-equation mapping, the weight of each
    /// variable (the number of equations in which it appears), and the priority
    /// of each equation (the number of variables in it). The
    /// variable-to-equation mapping is materialized as a vector of mutable
    /// slices, each of which points inside a provided backing vector. This
    /// approach greatly reduces the number of allocations.
    fn setup<'a>(
        &self,
        backing: &'a mut Vec<usize>,
    ) -> (Vec<&'a mut [usize]>, Vec<usize>, Vec<usize>) {
        let mut weight = vec![0; self.num_vars];
        let mut priority = vec![0; self.equations.len()];

        let mut total_vars = 0;
        for (i, equation) in self.equations.iter().enumerate() {
            priority[i] = equation.vars.len();
            total_vars += equation.vars.len();
            for &var in &equation.vars {
                weight[var as usize] += 1;
            }
        }

        backing.resize(total_vars, 0);
        let mut var_to_eq: Vec<&mut [usize]> = Vec::with_capacity(self.num_vars);

        backing.arbitrary_chunks_mut(&weight).for_each(|chunk| {
            var_to_eq.push(chunk);
        });

        let mut pos = vec![0usize; self.num_vars];
        for (i, equation) in self.equations.iter().enumerate() {
            for &var in &equation.vars {
                let var = var as usize;
                var_to_eq[var][pos[var]] = i;
                pos[var] += 1;
            }
        }

        (var_to_eq, weight, priority)
    }

    /// Solves the system using lazy Gaussian elimination.
    pub fn lazy_gaussian_elimination(&mut self) -> Result<Vec<W>> {
        let num_vars = self.num_vars;
        let num_equations = self.equations.len();

        if num_equations == 0 {
            return Ok(vec![W::ZERO; num_vars]);
        }

        let mut backing = vec![];
        let (var_to_eqs, mut weight, mut priority);
        (var_to_eqs, weight, priority) = self.setup(&mut backing);

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

        let mut dense = vec![];
        let mut solved = vec![];
        let mut pivots = vec![];

        let equations = &mut self.equations;
        let mut idle = bit_vec![true; num_vars];

        let mut remaining = equations.len();

        while remaining != 0 {
            if equation_list.is_empty() {
                let mut var = variables.pop().unwrap();
                while weight[var] == 0 {
                    var = variables.pop().unwrap()
                }
                idle.set(var, false);
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
                    let equation = unsafe { &*(&equations[first] as *const Modulo2Equation<W>) };

                    let pivot = equation
                        .vars
                        .iter()
                        .copied()
                        .find(|x| idle.get(*x as usize))
                        .expect("Missing expected idle variable in equation");
                    pivots.push(pivot as usize);
                    solved.push(first);
                    weight[pivot as usize] = 0;
                    var_to_eqs[pivot as usize]
                        .as_ref()
                        .iter()
                        .filter(|&&eq_idx| eq_idx != first)
                        .for_each(|&eq| {
                            equations[eq].add(equation);

                            priority[eq] -= 1;
                            if priority[eq] == 1 {
                                equation_list.push(eq)
                            }
                        });
                }
            }
        }

        // SAFETY: the usage of the method is safe, as the equations have the
        // right number of variables
        let mut dense_system = unsafe { Modulo2System::from_parts(num_vars, dense) };
        let mut solution = dense_system.gaussian_elimination()?;

        for i in 0..solved.len() {
            let eq = &equations[solved[i]];
            let pivot = pivots[i];
            assert!(solution[pivot] == W::ZERO);
            solution[pivot] = eq.c ^ Modulo2Equation::<W>::eval_vars(&eq.vars, &solution);
        }

        Ok(solution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_ptr() {
        let a = [0, 1, 3, 4];
        let b = [0, 2, 4, 5];
        let mut c = Vec::with_capacity(a.len() + b.len());

        let ra = a.as_ptr_range();
        let rb = b.as_ptr_range();
        let mut dst = c.as_mut_ptr();
        unsafe {
            dst = Modulo2Equation::<u32>::add_ptr(ra.start, ra.end, rb.start, rb.end, dst);
            assert_eq!(dst.offset_from(c.as_ptr()), 4);
            c.set_len(4);
            assert_eq!(c, vec![1, 2, 3, 5]);
        }
    }

    #[test]
    fn test_equation_builder() {
        let eq = unsafe { Modulo2Equation::from_parts(vec![0, 1, 2], 3_usize) };
        assert_eq!(eq.vars.len(), 3);
        assert_eq!(eq.vars.to_vec(), vec![0, 1, 2]);
    }

    #[test]
    fn test_equation_addition() {
        let mut eq0 = unsafe { Modulo2Equation::from_parts(vec![1, 4, 9], 3_usize) };
        let eq1 = unsafe { Modulo2Equation::from_parts(vec![1, 4, 10], 3_usize) };
        eq0.add(&eq1);
        assert_eq!(eq0.vars, vec![9, 10]);
        assert_eq!(eq0.c, 0);
    }

    #[test]
    fn test_system_one_equation() {
        let mut system = Modulo2System::<usize>::new(2);
        let eq = unsafe { Modulo2Equation::from_parts(vec![0], 3_usize) };
        system.push(eq);
        let solution = system.lazy_gaussian_elimination();
        assert!(solution.is_ok());
        assert!(system.check(&solution.unwrap()));
    }

    #[test]
    fn test_impossible_system() {
        let mut system = Modulo2System::<usize>::new(1);
        let eq0 = unsafe { Modulo2Equation::from_parts(vec![0], 0_usize) };
        system.push(eq0);
        let eq1 = unsafe { Modulo2Equation::from_parts(vec![0], 1_usize) };
        system.push(eq1);
        let solution = system.lazy_gaussian_elimination();
        assert!(solution.is_err());
    }

    #[test]
    fn test_redundant_system() {
        let mut system = Modulo2System::<usize>::new(1);
        let eq0 = unsafe { Modulo2Equation::from_parts(vec![0], 0_usize) };
        system.push(eq0);
        let eq1 = unsafe { Modulo2Equation::from_parts(vec![0], 0_usize) };
        system.push(eq1);
        let solution = system.lazy_gaussian_elimination();
        assert!(solution.is_ok());
        assert!(system.check(&solution.unwrap()));
    }

    #[test]
    fn test_small_system() {
        let mut system = Modulo2System::<usize>::new(11);
        let mut eq = unsafe { Modulo2Equation::from_parts(vec![1, 4, 10], 0) };
        system.push(eq);
        eq = unsafe { Modulo2Equation::from_parts(vec![1, 4, 9], 2) };
        system.push(eq);
        eq = unsafe { Modulo2Equation::from_parts(vec![0, 6, 8], 0) };
        system.push(eq);
        eq = unsafe { Modulo2Equation::from_parts(vec![0, 6, 9], 1) };
        system.push(eq);
        eq = unsafe { Modulo2Equation::from_parts(vec![2, 4, 8], 2) };
        system.push(eq);
        eq = unsafe { Modulo2Equation::from_parts(vec![2, 6, 10], 0) };
        system.push(eq);

        let solution = system.lazy_gaussian_elimination();
        assert!(solution.is_ok());
        assert!(system.check(&solution.unwrap()));
    }

    #[test]
    fn test_var_to_vec_builder() {
        let system = unsafe {
            Modulo2System::from_parts(
                11,
                vec![
                    Modulo2Equation::from_parts(vec![1, 4, 10], 0_usize),
                    Modulo2Equation::from_parts(vec![1, 4, 9], 1),
                    Modulo2Equation::from_parts(vec![0, 6, 8], 2),
                    Modulo2Equation::from_parts(vec![0, 6, 9], 3),
                    Modulo2Equation::from_parts(vec![2, 4, 8], 4),
                    Modulo2Equation::from_parts(vec![2, 6, 10], 5),
                ],
            )
        };
        let mut backing: Vec<usize> = vec![];
        let (var_to_eqs, _weight, _priority) = system.setup(&mut backing);
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
    }
}
