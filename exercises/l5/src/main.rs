use rand::seq::IndexedRandom;
use rand::Rng;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{self, Write};
use std::thread;
use std::time::Duration;

/**
* 增加DFS生成迷宫
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

/// sockets: [Up, Down, Left, Right]
fn dir_index(d: Direction) -> usize {
    match d {
        Direction::Up => 0,
        Direction::Down => 1,
        Direction::Left => 2,
        Direction::Right => 3,
    }
}

fn in_bounds(size: usize, x: i32, y: i32) -> bool {
    x >= 0 && y >= 0 && (x as usize) < size && (y as usize) < size
}

struct WFCManager {
    size: usize,
    grid: Vec<Vec<Vec<&'static str>>>,
    rules: HashMap<&'static str, HashMap<Direction, Vec<&'static str>>>,
    weights: HashMap<&'static str, u32>,
    /// ✅ 为预设路径做“socket 精确匹配过滤”用
    sockets: HashMap<&'static str, [u8; 4]>,
    history: Vec<Snapshot>,
}

impl WFCManager {
    fn new(
        size: usize,
        tiles: Vec<&'static str>,
        rules: HashMap<&'static str, HashMap<Direction, Vec<&'static str>>>,
        weights: HashMap<&'static str, u32>,
        sockets: HashMap<&'static str, [u8; 4]>,
    ) -> Self {
        Self {
            size,
            grid: vec![vec![tiles; size]; size],
            rules,
            weights,
            sockets,
            history: Vec::new(),
        }
    }

    // ============================================================
    // 传播：从多个 seed 一次性传播（边界先设完 -> 统一传播）
    // ============================================================
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
                if !in_bounds(self.size, nx, ny) {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);

                // allowed：当前格子的所有候选 tile -> 在 dir 方向允许的邻居集合并集
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

                // 候选减少：继续传播
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

    fn propagate(&mut self, x: usize, y: usize) -> Result<(), ()> {
        self.propagate_from_seeds(&[(x, y)])
    }

    // ============================================================
    // 终端原地刷新显示（光标回到左上角 + 清除到末尾）
    // ============================================================
    fn display_live(&self) {
        // 回到左上角
        print!("\x1B[H");
        // 清除从光标到屏幕末尾
        print!("\x1B[J");
        self.display();
        io::stdout().flush().ok();
    }

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

    // ============================================================
    // ✅ DFS 生成完美迷宫 + 选更长路径 -> 作为预设路径硬约束写回
    // ============================================================

    /// 入口/出口必须在边界上，本函数根据边界点计算“向内一步”的内部点
    fn step_inside_from_edge(&self, p: (usize, usize)) -> Result<(usize, usize), ()> {
        let (x, y) = p;
        if x == 0 {
            Ok((1, y))
        } else if x == self.size - 1 {
            Ok((self.size - 2, y))
        } else if y == 0 {
            Ok((x, 1))
        } else if y == self.size - 1 {
            Ok((x, self.size - 2))
        } else {
            Err(())
        }
    }

    /// 在内部区域 [inner_min..=inner_max] x [inner_min..=inner_max] 生成“完美迷宫”
    /// - 使用随机 DFS carving（递归回溯法的非递归写法）
    /// - 输出：邻接表 adj：每个内部格子与哪些格子打通通道
    fn generate_perfect_maze_dfs(
        &self,
        inner_min: usize,
        inner_max: usize,
        rng: &mut impl Rng,
    ) -> HashMap<(usize, usize), Vec<(usize, usize)>> {
        // 初始化邻接表
        let mut adj: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
        for y in inner_min..=inner_max {
            for x in inner_min..=inner_max {
                adj.insert((x, y), Vec::new());
            }
        }

        // 随机起点
        let start = (
            rng.random_range(inner_min..=inner_max),
            rng.random_range(inner_min..=inner_max),
        );

        let mut stack = vec![start];
        let mut visited: HashSet<(usize, usize)> = HashSet::new();
        visited.insert(start);

        // DFS carving：不断向未访问邻居前进；无路则回溯
        while let Some(&(cx, cy)) = stack.last() {
            let mut neighbors = Vec::new();

            let candidates = [
                (cx, cy.wrapping_sub(1)),
                (cx, cy + 1),
                (cx.wrapping_sub(1), cy),
                (cx + 1, cy),
            ];

            for (nx, ny) in candidates {
                if nx < inner_min || nx > inner_max || ny < inner_min || ny > inner_max {
                    continue;
                }
                if !visited.contains(&(nx, ny)) {
                    neighbors.push((nx, ny));
                }
            }

            if let Some(&next) = neighbors.choose(rng) {
                // 打通通道（双向）
                adj.get_mut(&(cx, cy)).unwrap().push(next);
                adj.get_mut(&next).unwrap().push((cx, cy));

                visited.insert(next);
                stack.push(next);
            } else {
                // 回溯
                stack.pop();
            }
        }

        adj
    }

    /// 在迷宫邻接表上：计算 start->goal 距离（BFS）
    fn maze_distance(
        adj: &HashMap<(usize, usize), Vec<(usize, usize)>>,
        start: (usize, usize),
        goal: (usize, usize),
    ) -> Option<usize> {
        let mut q = VecDeque::new();
        let mut dist: HashMap<(usize, usize), usize> = HashMap::new();

        q.push_back(start);
        dist.insert(start, 0);

        while let Some(cur) = q.pop_front() {
            if cur == goal {
                return dist.get(&cur).copied();
            }
            for &nxt in adj.get(&cur)? {
                if !dist.contains_key(&nxt) {
                    dist.insert(nxt, dist[&cur] + 1);
                    q.push_back(nxt);
                }
            }
        }
        None
    }

    /// 在迷宫邻接表上：提取 start->goal 路径（BFS parent 回溯）
    fn maze_path(
        adj: &HashMap<(usize, usize), Vec<(usize, usize)>>,
        start: (usize, usize),
        goal: (usize, usize),
    ) -> Option<Vec<(usize, usize)>> {
        let mut q = VecDeque::new();
        let mut parent: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        let mut visited: HashSet<(usize, usize)> = HashSet::new();

        q.push_back(start);
        visited.insert(start);

        while let Some(cur) = q.pop_front() {
            if cur == goal {
                break;
            }
            for &nxt in adj.get(&cur)? {
                if visited.insert(nxt) {
                    parent.insert(nxt, cur);
                    q.push_back(nxt);
                }
            }
        }

        if !visited.contains(&goal) {
            return None;
        }

        // 回溯
        let mut path = vec![goal];
        let mut cur = goal;
        while cur != start {
            cur = *parent.get(&cur)?;
            path.push(cur);
        }
        path.reverse();
        Some(path)
    }

    /// 将“预设路径”写回到 WFC：对路径上的每个格子计算 required sockets 并精确匹配过滤候选
    fn apply_path_as_hard_constraints(
        &mut self,
        full_path: &[(usize, usize)],
        entry: (usize, usize),
        exit: (usize, usize),
    ) -> Result<(), ()> {
        let path_set: HashSet<(usize, usize)> = full_path.iter().copied().collect();

        // 计算 required sockets：只向“路径内邻居”开口，避免分叉（更像单解路径）
        let mut required_map: HashMap<(usize, usize), [u8; 4]> = HashMap::new();
        for &(x, y) in full_path {
            let mut req = [0u8; 4];
            let neighbors = [
                (x, y.wrapping_sub(1), Direction::Up),
                (x, y + 1, Direction::Down),
                (x.wrapping_sub(1), y, Direction::Left),
                (x + 1, y, Direction::Right),
            ];
            for (nx, ny, dir) in neighbors {
                if nx < self.size && ny < self.size && path_set.contains(&(nx, ny)) {
                    req[dir_index(dir)] = 1;
                }
            }
            required_map.insert((x, y), req);
        }

        // 对路径上的内部格子写硬约束（入口/出口保持边界固定值，不强行改形状）
        for &(x, y) in full_path {
            if (x, y) == entry || (x, y) == exit {
                continue;
            }

            let req = *required_map.get(&(x, y)).ok_or(())?;

            // 精确匹配 socket 形状（保证路径结构不乱）
            let filtered: Vec<&'static str> = self.grid[y][x]
                .iter()
                .copied()
                .filter(|t| self.sockets.get(t).map(|s| *s == req).unwrap_or(false))
                .collect();

            if filtered.is_empty() {
                // 理论上你这套 tile_defs 覆盖了所有 0/1 组合中常见的通路形状，
                // 但如果这里空了，就是 tile 集合不足或已有约束冲突
                return Err(());
            }

            self.grid[y][x] = filtered;
            self.propagate(x, y)?; // 立即传播，尽早暴露矛盾
        }

        Ok(())
    }

    /// 对外总入口：预设迷宫式路径
    /// - attempts 越大，越可能挑到更长更曲折的路径（更像迷宫）
    fn preset_maze_like_path(&mut self, entry: (usize, usize), exit: (usize, usize), attempts: usize) -> Result<(), ()> {
        let start = self.step_inside_from_edge(entry)?;
        let goal = self.step_inside_from_edge(exit)?;

        let inner_min = 1usize;
        let inner_max = self.size - 2;

        let mut rng = rand::rng();
        let mut best_adj: Option<HashMap<(usize, usize), Vec<(usize, usize)>>> = None;
        let mut best_dist: usize = 0;

        for _ in 0..attempts {
            let adj = self.generate_perfect_maze_dfs(inner_min, inner_max, &mut rng);
            if let Some(d) = Self::maze_distance(&adj, start, goal) {
                if d > best_dist {
                    best_dist = d;
                    best_adj = Some(adj);
                }
            }
        }

        let best_adj = best_adj.ok_or(())?;
        let inner_path = Self::maze_path(&best_adj, start, goal).ok_or(())?;

        // 拼成完整路径：entry -> (start..goal) -> exit
        let mut full_path = Vec::with_capacity(inner_path.len() + 2);
        full_path.push(entry);
        full_path.extend(inner_path);
        full_path.push(exit);

        self.apply_path_as_hard_constraints(&full_path, entry, exit)?;
        Ok(())
    }

    // ============================================================
    // WFC 主过程：最小熵 -> 尝试 -> 传播 -> 回溯（实时显示）
    // ============================================================
    pub fn run_visualize(&mut self, every: usize, delay_ms: u64) {
        let mut step: usize = 0;

        // 初始显示一帧
        self.display_live();
        if delay_ms > 0 {
            thread::sleep(Duration::from_millis(delay_ms));
        }

        while let Some((x, y)) = self.find_min_entropy_coords() {
            let options = self.get_weighted_options(&self.grid[y][x]);
            let mut success = false;

            for &chosen in &options {
                // 保存快照，用于失败回溯
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

                    step += 1;
                    if every != 0 && step % every == 0 {
                        self.display_live();
                        if delay_ms > 0 {
                            thread::sleep(Duration::from_millis(delay_ms));
                        }
                    }
                    break;
                } else {
                    // 传播失败：回滚 + 移除尝试过的 tile
                    let last = self.history.pop().unwrap();
                    self.grid = last.grid;
                    self.grid[y][x].retain(|&t| t != chosen);

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
                // 更大一步回溯
                if self.history.is_empty() {
                    panic!("无法生成（回溯栈为空）。");
                }
                let last = self.history.pop().unwrap();
                self.grid = last.grid;
                self.grid[last.y][last.x].retain(|&t| t != last.tile_tried);

                step += 1;
                if every != 0 && step % every == 0 {
                    self.display_live();
                    if delay_ms > 0 {
                        thread::sleep(Duration::from_millis(delay_ms));
                    }
                }
            }
        }

        // 最终停在最终画面
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

        // 你的权重随机排序逻辑保留
        opts.sort_by_key(|&t| {
            let w = *self.weights.get(t).unwrap_or(&1);
            (rng.random::<f64>().powf(1.0 / w as f64) * 1000.0) as u32
        });

        opts.reverse();
        opts
    }
}

// ============================================================
// 随机入口/出口 + “门”图块选择
// ============================================================

fn collect_edge_points(size: usize, exclude_corners: bool) -> Vec<(usize, usize)> {
    let mut pts = Vec::new();

    for x in 0..size {
        pts.push((x, 0));
        pts.push((x, size - 1));
    }
    for y in 0..size {
        pts.push((0, y));
        pts.push((size - 1, y));
    }

    if exclude_corners {
        pts.retain(|&(x, y)| !((x == 0 || x == size - 1) && (y == 0 || y == size - 1)));
    }

    pts
}

fn pick_random_entry_exit(size: usize, rng: &mut impl Rng) -> ((usize, usize), (usize, usize)) {
    let edge_points = collect_edge_points(size, true);
    let entry = *edge_points.choose(rng).unwrap();
    let mut exit = *edge_points.choose(rng).unwrap();
    while exit == entry {
        exit = *edge_points.choose(rng).unwrap();
    }
    (entry, exit)
}

/// 上/下边界：用 "┃"；左/右边界：用 "━"
fn gate_tile_for_edge(size: usize, p: (usize, usize)) -> &'static str {
    let (x, y) = p;
    if y == 0 || y == size - 1 {
        "┃"
    } else if x == 0 || x == size - 1 {
        "━"
    } else {
        "━"
    }
}

fn main() {
    // 清屏一次（建立“原地刷新”画布）
    print!("\x1B[2J\x1B[H");
    io::stdout().flush().ok();

    // tile_defs: (tile_char, sockets[Up,Down,Left,Right], weight)
    let tile_defs: [(&'static str, [u8; 4], u32); 12] = [
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

    // 根据 sockets 自动推导 rules + weights + sockets_map
    let mut rules: HashMap<&'static str, HashMap<Direction, Vec<&'static str>>> = HashMap::new();
    let mut weights: HashMap<&'static str, u32> = HashMap::new();
    let mut sockets_map: HashMap<&'static str, [u8; 4]> = HashMap::new();

    let all_names: Vec<&'static str> = tile_defs.iter().map(|(n, _, _)| *n).collect();

    for (name, sockets, weight) in &tile_defs {
        weights.insert(*name, *weight);
        sockets_map.insert(*name, *sockets);

        let mut dir_map: HashMap<Direction, Vec<&'static str>> = HashMap::new();
        for (n_other, s_other, _) in &tile_defs {
            // socket 匹配推导邻居合法性
            if sockets[0] == s_other[1] {
                dir_map.entry(Direction::Up).or_default().push(*n_other);
            }
            if sockets[1] == s_other[0] {
                dir_map.entry(Direction::Down).or_default().push(*n_other);
            }
            if sockets[2] == s_other[3] {
                dir_map.entry(Direction::Left).or_default().push(*n_other);
            }
            if sockets[3] == s_other[2] {
                dir_map.entry(Direction::Right).or_default().push(*n_other);
            }
        }
        rules.insert(*name, dir_map);
    }

    let size = 30;
    let mut wfc = WFCManager::new(size, all_names, rules, weights, sockets_map);

    // -----------------------------
    // 1) 随机入口/出口（四边任意组合）
    // -----------------------------
    let mut rng = rand::rng();
    let (entry, exit) = pick_random_entry_exit(size, &mut rng);
    let entry_tile = gate_tile_for_edge(size, entry);
    let exit_tile = gate_tile_for_edge(size, exit);

    // -----------------------------
    // 2) 边界先全部设完（不传播）并收集 seeds
    // -----------------------------
    let mut seeds: Vec<(usize, usize)> = Vec::new();
    for y in 0..size {
        for x in 0..size {
            let is_edge = y == 0 || y == size - 1 || x == 0 || x == size - 1;
            if !is_edge {
                continue;
            }
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

    // ✅ 统一传播（只做一次）
    wfc.propagate_from_seeds(&seeds).expect("边界统一传播出现矛盾");

    // 打印入口出口信息（注意：这会占用终端顶部一行；不想影响画面可注释掉）
    println!("Entry: {:?} tile={}  Exit: {:?} tile={}", entry, entry_tile, exit, exit_tile);

    // -----------------------------
    // 3) 使用 DFS 生成完美迷宫 -> 选更长解路径 -> 写成硬约束
    // -----------------------------
    // attempts 越大，越迷宫（更曲折更长），但也更费时
    wfc.preset_maze_like_path(entry, exit, 40)
        .expect("预设迷宫路径失败（约束冲突/形状不足）");

    // -----------------------------
    // 4) 运行 WFC 并原地刷新观察
    // -----------------------------
    // every=1 每一步都刷；delay_ms 控制速度（例如 10~30 看得更清楚）
    wfc.run_visualize(1, 20);
}
