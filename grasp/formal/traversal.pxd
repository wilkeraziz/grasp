from grasp.formal.hg cimport Hypergraph


cpdef tuple top_down_left_right(Hypergraph forest, tuple acyclic_derivation, bint terminal_only=?)

cpdef tuple bracketing(Hypergraph forest, tuple acyclic_derivation)

cpdef str bracketed_string(Hypergraph forest, tuple acyclic_derivation,
                           str bos=?, str ios=?, str eos=?,
                           bint label_bos=?, bint label_eos=?,
                           str bos_position=?, str ios_position=?, str eos_position=?)

cpdef str yield_string(Hypergraph forest, tuple acyclic_derivation)