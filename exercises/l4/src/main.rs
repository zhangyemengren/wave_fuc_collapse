use rand::seq::IndexedRandom;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::thread;
use std::time::Duration;

/**
* 图块根规则 复杂度增加
* 增加权重
* 增加回溯
*/

#[derive(Clone)]
struct Snapshot {
    grid: Vec<Vec<Vec<&'static str>>>,
    x: usize,
    y: usize,
    tile_tried: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

struct WFCManager {
    size: usize,
    grid: Vec<Vec<Vec<&'static str>>>,
    rules: HashMap<&'static str, HashMap<Direction, Vec<&'static str>>>,
    weights: HashMap<&'static str, u32>,
    history: Vec<Snapshot>,
}

impl WFCManager {
    fn new(
        size: usize,
        tiles: Vec<&'static str>,
        rules: HashMap<&'static str, HashMap<Direction, Vec<&'static str>>>,
        weights: HashMap<&'static str, u32>,
    ) -> Self {
        Self {
            size,
            grid: vec![vec![tiles; size]; size],
            rules,
            weights,
            history: Vec::new(),
        }
    }

    // -----------------------------------------
    // ✅ 新增：从多个 seed 一次性传播（用于“边界先设完再传播”）
    // -----------------------------------------
    //
    // 之前你是：每设一个边界格子就 propagate 一次，传播顺序会导致“过早收敛”，
    // 后面再设另一个边界点时更容易冲突。
    //
    // 现在改成：边界全部固定完 -> seeds 收集完 -> 一次性 propagate_from_seeds(seeds)
    fn propagate_from_seeds(&mut self, seeds: &[(usize, usize)]) -> Result<(), ()> {
        let mut stack = seeds.to_vec();

        while let Some((cx, cy)) = stack.pop() {
            let neighbors = [
                (0, -1, Direction::Up),
                (0, 1, Direction::Down),
                (-1, 0, Direction::Left),
                (1, 0, Direction::Right),
            ];

            for (dx, dy, dir) in neighbors {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;

                if nx < 0 || ny < 0 || nx >= self.size as i32 || ny >= self.size as i32 {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);

                // allowed：由当前格子的所有候选 tile 推导出来的“邻格允许集合”
                let mut allowed = HashSet::new();
                for &tile in &self.grid[cy][cx] {
                    if let Some(valid) = self.rules.get(tile).and_then(|r| r.get(&dir)) {
                        for &v in valid {
                            allowed.insert(v);
                        }
                    }
                }

                // 用 allowed 过滤邻格候选
                let cell = &mut self.grid[ny][nx];
                let old_len = cell.len();
                cell.retain(|t| allowed.contains(t));

                // 候选变少：继续把邻格入栈传播
                if cell.len() < old_len {
                    if cell.is_empty() {
                        return Err(());
                    }
                    stack.push((nx, ny));
                }
            }
        }

        Ok(())
    }

    // 兼容原先的单点传播（内部仍然复用 seeds 版本）
    fn propagate(&mut self, x: usize, y: usize) -> Result<(), ()> {
        self.propagate_from_seeds(&[(x, y)])
    }

    // ------------------------------------------------
    // ✅ 新增：终端实时刷新显示（清屏+打印+flush）
    // ------------------------------------------------
    fn display_live(&self) {
        // 1) 把光标移动到左上角
        print!("\x1B[H");
        // 2) 清除从光标到屏幕末尾（避免残留旧内容）
        print!("\x1B[J");

        // 3) 打印当前网格
        self.display();

        // 4) 强制刷新 stdout，避免缓冲导致“堆叠”
        std::io::stdout().flush().ok();
    }

    // 原始 display（不清屏，直接输出最终结果也可用）
    fn display(&self) {
        for row in &self.grid {
            for cell in row {
                if cell.len() == 1 {
                    print!("{}", cell[0]);
                } else {
                    print!("·");
                }
            }
            println!();
        }
    }

    // -----------------------------------------
    // 运行：实时可视化版本
    // -----------------------------------------
    //
    // every: 每多少次“成功折叠/回溯”刷新一次（1=每步都刷）
    // delay_ms: 每次刷新后暂停多少毫秒（0=不暂停）
    pub fn run_visualize(&mut self, every: usize, delay_ms: u64) {
        let mut step: usize = 0;

        // 初始画面先打印一次
        self.display_live();
        if delay_ms > 0 {
            thread::sleep(Duration::from_millis(delay_ms));
        }

        while let Some((x, y)) = self.find_min_entropy_coords() {
            let options = self.get_weighted_options(&self.grid[y][x]);
            let mut success = false;

            for &chosen in &options {
                // 快照用于回溯
                self.history.push(Snapshot {
                    grid: self.grid.clone(),
                    x,
                    y,
                    tile_tried: chosen,
                });

                // 折叠该格子
                self.grid[y][x] = vec![chosen];

                // 传播
                if self.propagate(x, y).is_ok() {
                    success = true;

                    // ✅ 成功折叠后刷新
                    step += 1;
                    if every != 0 && step % every == 0 {
                        self.display_live();
                        if delay_ms > 0 {
                            thread::sleep(Duration::from_millis(delay_ms));
                        }
                    }
                    break;
                } else {
                    // 传播失败：回滚快照，并把这个 chosen 从该格候选里移除
                    let last = self.history.pop().unwrap();
                    self.grid = last.grid;
                    self.grid[y][x].retain(|&t| t != chosen);

                    // ✅ 回溯也刷新（可观察“试错”）
                    step += 1;
                    if every != 0 && step % every == 0 {
                        self.display_live();
                        if delay_ms > 0 {
                            thread::sleep(Duration::from_millis(delay_ms));
                        }
                    }
                }
            }

            if !success {
                // 所有候选都失败：更大一步回溯
                if self.history.is_empty() {
                    panic!("无法生成符合约束的迷宫（回溯栈为空）。");
                }

                let last = self.history.pop().unwrap();
                self.grid = last.grid;
                self.grid[last.y][last.x].retain(|&t| t != last.tile_tried);

                // ✅ 大回溯也刷新
                step += 1;
                if every != 0 && step % every == 0 {
                    self.display_live();
                    if delay_ms > 0 {
                        thread::sleep(Duration::from_millis(delay_ms));
                    }
                }
            }
        }

        // 最终再刷一次，确保停在最终画面
        self.display_live();
    }

    fn find_min_entropy_coords(&self) -> Option<(usize, usize)> {
        let mut min_e = f64::MAX;
        let mut candidates = Vec::new();

        for y in 0..self.size {
            for x in 0..self.size {
                if self.grid[y][x].len() > 1 {
                    let e = self.get_entropy(x, y);
                    if e < min_e - 0.001 {
                        min_e = e;
                        candidates = vec![(x, y)];
                    } else if (e - min_e).abs() < 0.001 {
                        candidates.push((x, y));
                    }
                }
            }
        }

        candidates.choose(&mut rand::rng()).copied()
    }

    fn get_entropy(&self, x: usize, y: usize) -> f64 {
        let sum_w: f64 = self.grid[y][x]
            .iter()
            .map(|&t| *self.weights.get(t).unwrap() as f64)
            .sum();
        let sum_w_log_w: f64 = self.grid[y][x]
            .iter()
            .map(|&t| {
                let w = *self.weights.get(t).unwrap() as f64;
                w * w.ln()
            })
            .sum();
        sum_w.ln() - (sum_w_log_w / sum_w)
    }

    fn get_weighted_options(&self, options: &[&'static str]) -> Vec<&'static str> {
        let mut opts = options.to_vec();
        let mut rng = rand::rng();

        opts.sort_by_key(|&t| {
            let w = *self.weights.get(t).unwrap_or(&1);
            // 你原来的“权重随机排序”逻辑
            (rng.random::<f64>().powf(1.0 / w as f64) * 1000.0) as u32
        });

        opts.reverse();
        opts
    }
}


/// 收集所有边界点（可选择排除四个角，避免入口卡在角上）
fn collect_edge_points(size: usize, exclude_corners: bool) -> Vec<(usize, usize)> {
    let mut pts = Vec::new();

    // 上/下边
    for x in 0..size {
        pts.push((x, 0));
        pts.push((x, size - 1));
    }
    // 左/右边
    for y in 0..size {
        pts.push((0, y));
        pts.push((size - 1, y));
    }

    if exclude_corners {
        pts.retain(|&(x, y)| !((x == 0 || x == size - 1) && (y == 0 || y == size - 1)));
    }

    pts
}

/// 从边界点随机挑入口和出口（保证不同）
fn pick_random_entry_exit(size: usize, rng: &mut impl rand::Rng) -> ((usize, usize), (usize, usize)) {
    let edge_points = collect_edge_points(size, true); // true=排除角
    let entry = *edge_points.choose(rng).unwrap();

    let mut exit = *edge_points.choose(rng).unwrap();
    while exit == entry {
        exit = *edge_points.choose(rng).unwrap();
    }
    (entry, exit)
}

/// 根据入口/出口所在边返回“门”的图块：
/// - 在上/下边界：应该竖向连进来 -> "┃"
/// - 在左/右边界：应该横向连进来 -> "━"
fn gate_tile_for_edge(size: usize, p: (usize, usize)) -> &'static str {
    let (x, y) = p;
    if y == 0 || y == size - 1 {
        "┃"
    } else if x == 0 || x == size - 1 {
        "━"
    } else {
        // 理论不会发生：入口/出口必须在边界
        "━"
    }
}


fn main() {
    print!("\x1B[2J\x1B[H");
    std::io::stdout().flush().ok();
    let tile_defs = [
        ("█", [0, 0, 0, 0], 40),
        ("┃", [1, 1, 0, 0], 20),
        ("━", [0, 0, 1, 1], 20),
        ("┏", [0, 1, 0, 1], 15),
        ("┓", [0, 1, 1, 0], 15),
        ("┗", [1, 0, 0, 1], 15),
        ("┛", [1, 0, 1, 0], 15),
        ("┣", [1, 1, 0, 1], 5),
        ("┫", [1, 1, 1, 0], 5),
        ("┳", [0, 1, 1, 1], 5),
        ("┻", [1, 0, 1, 1], 5),
        ("╋", [1, 1, 1, 1], 2),
    ];

    let mut rules = HashMap::new();
    let mut weights = HashMap::new();
    let all_names: Vec<&str> = tile_defs.iter().map(|(n, _, _)| *n).collect();

    // 自动推导 rules
    for (name, sockets, weight) in &tile_defs {
        weights.insert(*name, *weight);
        let mut dir_map = HashMap::new();

        for (n_other, s_other, _) in &tile_defs {
            if sockets[0] == s_other[1] {
                dir_map.entry(Direction::Up).or_insert(vec![]).push(*n_other);
            }
            if sockets[1] == s_other[0] {
                dir_map
                    .entry(Direction::Down)
                    .or_insert(vec![])
                    .push(*n_other);
            }
            if sockets[2] == s_other[3] {
                dir_map
                    .entry(Direction::Left)
                    .or_insert(vec![])
                    .push(*n_other);
            }
            if sockets[3] == s_other[2] {
                dir_map
                    .entry(Direction::Right)
                    .or_insert(vec![])
                    .push(*n_other);
            }
        }
        rules.insert(*name, dir_map);
    }

    let size = 20;
    let mut wfc = WFCManager::new(size, all_names, rules, weights);

    // -----------------------------
    // ✅ 边界先全部设完（不传播）
    // -----------------------------
    let mut rng = rand::rng();
    let (entry, exit) = pick_random_entry_exit(size, &mut rng);

    let entry_tile = gate_tile_for_edge(size, entry);
    let exit_tile  = gate_tile_for_edge(size, exit);

    // 边界先全部设完（不传播），并收集 seeds，最后统一传播
    let mut seeds: Vec<(usize, usize)> = Vec::new();

    for y in 0..size {
        for x in 0..size {
            let is_edge = y == 0 || y == size - 1 || x == 0 || x == size - 1;
            if !is_edge { continue; }

            if (x, y) == entry {
                wfc.grid[y][x] = vec![entry_tile];
            } else if (x, y) == exit {
                wfc.grid[y][x] = vec![exit_tile];
            } else {
                wfc.grid[y][x] = vec!["█"];
            }

            seeds.push((x, y));
        }
    }

    // ✅ 边界全部设完后统一传播
    wfc.propagate_from_seeds(&seeds).expect("边界统一传播出现矛盾");

    // （可选）打印入口出口信息
    println!("Entry: {:?} tile={}  Exit: {:?} tile={}", entry, entry_tile, exit, exit_tile);

    // -----------------------------
    // 运行 WFC：实时观察
    // -----------------------------
    println!("生成中（实时显示）...");
    // every=1 表示每一步都刷；delay_ms=20 可以让你看得清楚些
    wfc.run_visualize(1, 20);
}
