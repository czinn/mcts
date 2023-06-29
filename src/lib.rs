/// This is a library for Monte Carlo tree search.
///
/// See `examples/counting_game.rs` for an example of usage.
extern crate crossbeam;
extern crate smallvec;

mod atomics;
mod search_tree;
pub mod transposition_table;
pub mod tree_policy;

pub use search_tree::*;
use transposition_table::*;
use tree_policy::*;

use async_scoped::TokioScope as Scope;
use async_trait::async_trait;
use atomics::*;
use std::sync::Arc;
use std::time::Duration;

pub trait MCTS: Sized + Sync + Send {
    type State: GameState + Sync + Send;
    type Eval: Evaluator<Self> + Sync + Send;
    type TreePolicy: TreePolicy<Self> + Sync + Send;
    type NodeData: Default + Sync + Send;
    type TranspositionTable: TranspositionTable<Self> + Sync + Send;
    type ExtraThreadData: Sync + Send;

    fn virtual_loss(&self) -> i64 {
        0
    }
    fn visits_before_expansion(&self) -> u64 {
        1
    }
    fn node_limit(&self) -> usize {
        std::usize::MAX
    }
    fn select_child_after_search<'a>(&self, children: &'a [MoveInfo<Self>]) -> &'a MoveInfo<Self> {
        children
            .into_iter()
            .max_by_key(|child| child.visits())
            .unwrap()
    }
    /// `playout` panics when this length is exceeded. Defaults to one million.
    fn max_playout_length(&self) -> usize {
        1_000_000
    }
    fn on_backpropagation(&self, _evaln: &StateEvaluation<Self>, _handle: SearchHandle<Self>) {}
    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        if std::mem::size_of::<Self::TranspositionTable>() == 0 {
            CycleBehaviour::Ignore
        } else {
            CycleBehaviour::PanicWhenCycleDetected
        }
    }
}

pub struct ThreadData<Spec: MCTS> {
    pub policy_data: TreePolicyThreadData<Spec>,
    pub extra_data: Spec::ExtraThreadData,
}

impl<Spec: MCTS> Default for ThreadData<Spec>
where
    TreePolicyThreadData<Spec>: Default,
    Spec::ExtraThreadData: Default,
{
    fn default() -> Self {
        Self {
            policy_data: Default::default(),
            extra_data: Default::default(),
        }
    }
}

pub type MoveEvaluation<Spec> = <<Spec as MCTS>::TreePolicy as TreePolicy<Spec>>::MoveEvaluation;
pub type StateEvaluation<Spec> = <<Spec as MCTS>::Eval as Evaluator<Spec>>::StateEvaluation;
pub type Move<Spec> = <<Spec as MCTS>::State as GameState>::Move;
pub type MoveList<Spec> = <<Spec as MCTS>::State as GameState>::MoveList;
pub type Player<Spec> = <<Spec as MCTS>::State as GameState>::Player;
pub type TreePolicyThreadData<Spec> =
    <<Spec as MCTS>::TreePolicy as TreePolicy<Spec>>::ThreadLocalData;

pub trait GameState: Clone {
    type Move: Sync + Send + Clone;
    type Player: Sync + Send;
    type MoveList: std::iter::IntoIterator<Item = Self::Move> + Sync + Send;

    fn current_player(&self) -> Self::Player;
    fn available_moves(&self) -> Self::MoveList;
    fn make_move(&mut self, mov: &Self::Move);
}

#[async_trait]
pub trait Evaluator<Spec: MCTS>: Sync {
    type StateEvaluation: Sync + Send;

    async fn evaluate_new_state(
        &self,
        state: &Spec::State,
        moves: &MoveList<Spec>,
    ) -> (Vec<MoveEvaluation<Spec>>, Self::StateEvaluation);

    fn evaluate_existing_state(
        &self,
        state: &Spec::State,
        existing_evaln: &Self::StateEvaluation,
        handle: SearchHandle<Spec>,
    ) -> Self::StateEvaluation;

    fn interpret_evaluation_for_player(
        &self,
        evaluation: &Self::StateEvaluation,
        player: &Player<Spec>,
    ) -> i64;
}

pub struct MCTSManager<Spec: MCTS> {
    search_tree: SearchTree<Spec>,
    // thread local data when we have no asynchronous workers
    single_threaded_tld: Option<ThreadData<Spec>>,
    print_on_playout_error: bool,
}

impl<Spec: MCTS> MCTSManager<Spec>
where
    ThreadData<Spec>: Default,
{
    pub async fn new(
        state: Spec::State,
        manager: Spec,
        eval: Spec::Eval,
        tree_policy: Spec::TreePolicy,
        table: Spec::TranspositionTable,
    ) -> Self {
        let search_tree = SearchTree::new(state, manager, tree_policy, eval, table).await;
        let single_threaded_tld = None;
        Self {
            search_tree,
            single_threaded_tld,
            print_on_playout_error: true,
        }
    }

    pub fn print_on_playout_error(&mut self, v: bool) -> &mut Self {
        self.print_on_playout_error = v;
        self
    }

    pub async fn playout(&mut self) {
        // Avoid overhead of thread creation
        if self.single_threaded_tld.is_none() {
            self.single_threaded_tld = Some(Default::default());
        }
        self.search_tree
            .playout(self.single_threaded_tld.as_mut().unwrap())
            .await;
    }
    pub async fn playout_until<Predicate: FnMut() -> bool>(&mut self, mut pred: Predicate) {
        while !pred() {
            self.playout().await;
        }
    }
    pub async fn playout_n(&mut self, n: u64) {
        for _ in 0..n {
            self.playout().await;
        }
    }
    fn spawn_worker_thread<'a>(&'a self, scope: &mut Scope<'a, ()>, stop_signal: Arc<AtomicBool>) {
        let search_tree = &self.search_tree;
        let print_on_playout_error = self.print_on_playout_error;
        scope.spawn(async move {
            let mut tld = Default::default();
            loop {
                if stop_signal.load(Ordering::SeqCst) {
                    break;
                }
                if !search_tree.playout(&mut tld).await {
                    if print_on_playout_error {
                        eprintln!(
                            "Node limit of {} reached. Halting search.",
                            search_tree.spec().node_limit()
                        );
                    }
                    break;
                }
                tokio::task::yield_now().await;
            }
        });
    }
    fn playout_parallel_async<'a>(
        &'a self,
        scope: &mut Scope<'a, ()>,
        num_threads: usize,
    ) -> AsyncSearch {
        assert!(num_threads != 0);
        let stop_signal = Arc::new(AtomicBool::new(false));
        for _i in 0..num_threads {
            let stop_signal = stop_signal.clone();
            self.spawn_worker_thread(scope, stop_signal);
        }
        AsyncSearch { stop_signal }
    }
    pub fn playout_parallel_for(&mut self, duration: Duration, num_threads: usize) {
        Scope::scope_and_block(|scope| {
            let search = self.playout_parallel_async(scope, num_threads);
            scope.spawn(async {
                tokio::time::sleep(duration).await;
                search.halt();
            });
        });
    }
    pub fn playout_n_parallel(&mut self, n: u32, num_threads: usize) {
        if n == 0 {
            return;
        }
        assert!(num_threads != 0);
        let counter = AtomicIsize::new(n as isize);
        let search_tree = &self.search_tree;
        Scope::scope_and_block(|scope| {
            for _ in 0..num_threads {
                scope.spawn(async {
                    let mut tld = Default::default();
                    loop {
                        let count = counter.fetch_sub(1, Ordering::SeqCst);
                        if count <= 0 {
                            break;
                        }
                        search_tree.playout(&mut tld).await;
                    }
                });
            }
        });
    }
    pub fn principal_variation_info(&self, num_moves: usize) -> Vec<MoveInfoHandle<Spec>> {
        self.search_tree.principal_variation(num_moves)
    }
    pub fn principal_variation(&self, num_moves: usize) -> Vec<Move<Spec>> {
        self.search_tree
            .principal_variation(num_moves)
            .into_iter()
            .map(|x| x.get_move())
            .map(|x| x.clone())
            .collect()
    }
    pub fn principal_variation_states(&self, num_moves: usize) -> Vec<Spec::State> {
        let moves = self.principal_variation(num_moves);
        let mut states = vec![self.search_tree.root_state().clone()];
        for mov in moves {
            let mut state = states[states.len() - 1].clone();
            state.make_move(&mov);
            states.push(state);
        }
        states
    }
    pub fn tree(&self) -> &SearchTree<Spec> {
        &self.search_tree
    }
    pub fn best_move(&self) -> Option<Move<Spec>> {
        self.principal_variation(1).get(0).map(|x| x.clone())
    }
    pub fn perf_test<F>(&mut self, num_threads: usize, mut f: F)
    where
        F: FnMut(usize) + Send,
    {
        Scope::scope_and_block(|scope| {
            let search = self.playout_parallel_async(scope, num_threads);
            scope.spawn(async {
                for _ in 0..3 {
                    let n1 = self.search_tree.root_node().visits();
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    let n2 = self.search_tree.root_node().visits();
                    let diff = if n2 > n1 { n2 - n1 } else { 0 };
                    f(diff);
                }
                search.halt();
            });
        });
    }
    pub fn perf_test_to_stderr(&mut self, num_threads: usize) {
        self.perf_test(num_threads, |x| {
            eprintln!("{} nodes/sec", thousands_separate(x))
        });
    }
    pub async fn reset(self) -> Self {
        Self {
            search_tree: self.search_tree.reset().await,
            print_on_playout_error: self.print_on_playout_error,
            single_threaded_tld: None,
        }
    }
}

// https://stackoverflow.com/questions/26998485/rust-print-format-number-with-thousand-separator
fn thousands_separate(x: usize) -> String {
    let s = format!("{}", x);
    let bytes: Vec<_> = s.bytes().rev().collect();
    let chunks: Vec<_> = bytes
        .chunks(3)
        .map(|chunk| String::from_utf8(chunk.to_vec()).unwrap())
        .collect();
    let result: Vec<_> = chunks.join(",").bytes().rev().collect();
    String::from_utf8(result).unwrap()
}

#[must_use]
pub struct AsyncSearch {
    pub stop_signal: Arc<AtomicBool>,
}

impl AsyncSearch {
    pub fn halt(self) {
        self.stop_signal.store(true, Ordering::SeqCst);
    }
}

pub enum CycleBehaviour<Spec: MCTS> {
    Ignore,
    UseCurrentEvalWhenCycleDetected,
    PanicWhenCycleDetected,
    UseThisEvalWhenCycleDetected(StateEvaluation<Spec>),
}
