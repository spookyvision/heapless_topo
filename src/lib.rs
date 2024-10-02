#![cfg_attr(all(not(test), not(feature = "std")), no_std)]
//! A bare-bones `no_std` implementation of topological sort, using `heapless` for storage.
//!
//! This crate assumes your main graph data is stored elsewhere and you only create a `Graph` temporarily
//! with the express purpose of doing toposort. Hence the design decisions:
//! - only store edge information, no node payload
//! - consume `self` on sort.
//!
//! # Capacity requirements
//! The implementation uses one single const generic for all temporary data structures. In the pathological case
//! it requires (number of graph edges) + 1, so be mindful of that: a toposort of e.g. `[(0,1), (1,2)]` is `[0,1,2]`.
//!
//! # Usage
//!
//! ```
//! use heapless_topo::{Graph, Edge};
//! const CAPACITY: usize = 8;
//! // or `new_with_edges` if you have a `Vec<Edge>` already
//! let mut graph = Graph::<CAPACITY>::new();
//! graph.insert_edge(Edge::from((1,2)));
//! graph.insert_edge(Edge::from((0,1)));
//! let sorted = graph.into_topo_sorted();
//! let expected = [0,1,2].as_slice().try_into().unwrap();
//! assert_eq!(Ok(expected), sorted);
//! ```
//!
//! # Crate features
//! - `std` for `#[derive(Debug)]`
//! - `defmt-03` for `#[derive(Format)]` using `defmt` v0.3
//!

use heapless::{FnvIndexSet, Vec};

#[cfg_attr(any(test, feature = "std"), derive(Debug))]
#[cfg_attr(feature = "defmt-03", derive(defmt::Format))]
#[derive(PartialEq, Eq)]
pub enum Error {
    Cycle,
    OverCapacity,
}

/// Graph edge
#[cfg_attr(any(test, feature = "std"), derive(Debug))]
#[cfg_attr(feature = "defmt-03", derive(defmt::Format))]
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct Edge {
    from: usize,
    to: usize,
}

impl From<(usize, usize)> for Edge {
    fn from(value: (usize, usize)) -> Self {
        Self {
            from: value.0,
            to: value.1,
        }
    }
}

/// payload-agnostic Graph (pure edge data)
#[cfg_attr(any(test, feature = "std"), derive(Debug))]
#[cfg_attr(feature = "defmt-03", derive(defmt::Format))]
#[derive(Default, PartialEq, Eq, Clone)]

pub struct Graph<const EDGES: usize> {
    edges: Vec<Edge, EDGES>,
}

impl<const EDGES: usize> Graph<EDGES> {
    /// Create a new, empty graph
    pub fn new() -> Self {
        Self { edges: Vec::new() }
    }

    /// Create a new graph from existing edge data
    pub fn new_with_edges(edges: Vec<Edge, EDGES>) -> Self {
        Self { edges }
    }

    /// Insert an edge into the graph. No duplicate check is performed.
    /// Returns `Error::OverCapacity` if full.
    pub fn insert_edge(&mut self, edge: Edge) -> Result<(), Error> {
        self.edges.push(edge).map_err(|_| Error::OverCapacity)
    }

    /// compute topological sort, consuming self.
    pub fn into_topo_sorted(self) -> Result<Vec<usize, EDGES>, Error> {
        let mut res = Vec::new();
        // compute a list of starting nodes, i.e. nodes with no incoming edges
        //
        // reuse EDGES size here since it's an upper bound.
        // nb: in dense graphs this is wasteful.
        let mut starting_nodes: FnvIndexSet<usize, EDGES> = FnvIndexSet::new();

        let mut edges = self.edges;
        // for all edges, assume they go from a starting node
        for edge in &edges {
            starting_nodes
                .insert(edge.from)
                .map_err(|_| Error::OverCapacity)?;
        }

        // now remove all nodes that do have an incoming edge
        for edge in &edges {
            if starting_nodes.contains(&edge.to) {
                starting_nodes.remove(&edge.to);
            }
        }

        // Kahn's algorithm
        // https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
        // L (here: `res`) ← Empty list that will contain the sorted elements
        // S (here: `starting_nodes`) ← Set of all nodes with no incoming edge

        while !starting_nodes.is_empty() {
            // 1. remove a node n from S
            // unwrap safety: we just checked !is_empty
            let node = *starting_nodes.first().unwrap();
            starting_nodes.remove(&node);

            // add N to L
            res.push(node).map_err(|_| Error::OverCapacity)?;

            // for each node m with an edge e from n to m, do
            // remove edge e from the graph

            // keep track of edges that have become starting
            let mut starting_edges: Vec<bool, EDGES> = Vec::new();
            // fill with default (false)
            starting_edges.resize_default(EDGES).unwrap();
            for (idx, edge) in edges.iter().enumerate() {
                if edge.from == node {
                    starting_edges[idx] = true;
                    // check if m has other incoming edges, if not, add m to S
                    let mut m_has_become_starting = true;
                    for check_edge in &edges {
                        if check_edge.to == edge.to && check_edge.from != edge.from {
                            m_has_become_starting = false;
                            break;
                        }
                    }
                    if m_has_become_starting {
                        starting_nodes
                            .insert(edge.to)
                            .map_err(|_| Error::OverCapacity)?;
                    }
                }
            }

            // retain all edges that have *not* been flagged as starting
            // unwrap safety: number of starting edges <= total number of edges,
            // hence the iterator never gets exhausted
            let mut it = starting_edges.into_iter();
            edges.retain(|_| !it.next().unwrap());
        }
        if edges.is_empty() {
            Ok(res)
        } else {
            Err(Error::Cycle)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const CAPACITY: usize = 32;
    #[test]
    fn ok() {
        // the first 4 edges imply the only possible topological sorting is 1,2,3,4,5
        let edge_data = [(1, 2), (2, 3), (3, 4), (4, 5), (3, 5), (1, 5)];
        let mut edges: heapless::Vec<Edge, CAPACITY> = heapless::Vec::new();

        // construct graph from a reversed iter to reduce likelihood of accidental success
        for edge in edge_data.into_iter().rev() {
            edges
                .push(edge.into())
                .expect("bug in test case: edge vec over capacity");
        }
        let graph = Graph::new_with_edges(edges);
        let res = graph.into_topo_sorted();
        let expected = [1, 2, 3, 4, 5].as_slice().try_into().unwrap();
        assert_eq!(Ok(expected), res);
    }

    #[test]
    fn ok_with_push() {
        // the first 4 edges imply the only possible topological sorting is 1,2,3,4,5
        let edge_data = [(1, 2), (2, 3), (3, 4), (4, 5), (3, 5), (1, 5)];
        let mut graph = Graph::<CAPACITY>::new();
        for edge in edge_data.into_iter() {
            graph.insert_edge(edge.into()).unwrap();
        }
        let res = graph.into_topo_sorted();
        let expected = [1, 2, 3, 4, 5].as_slice().try_into().unwrap();
        assert_eq!(Ok(expected), res);
    }

    #[test]
    fn err_too_many_edges() {
        let mut graph = Graph::<1>::new();
        assert_eq!(Ok(()), graph.insert_edge((1, 2).into()));
        assert_eq!(Err(Error::OverCapacity), graph.insert_edge((2, 3).into()));
    }

    #[test]
    fn err_num_nodes_greater_than_num_edges() {
        let mut graph = Graph::<2>::new();
        assert_eq!(Ok(()), graph.insert_edge((1, 2).into()));
        assert_eq!(Ok(()), graph.insert_edge((0, 1).into()));
        assert_eq!(Err(Error::OverCapacity), graph.into_topo_sorted());
    }

    #[test]
    fn err_cycles() {
        let edge_data = [(1, 2), (2, 3), (3, 4), (4, 5), (3, 5), (1, 5), (5, 1)];
        let mut edges: heapless::Vec<Edge, CAPACITY> = heapless::Vec::new();

        for edge in edge_data {
            edges
                .push(edge.into())
                .expect("bug in test case: edge vec over capacity");
        }
        let graph = Graph::new_with_edges(edges);
        let res = graph.into_topo_sorted();
        assert_eq!(Err(Error::Cycle), res);
    }
}
