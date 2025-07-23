#[macro_export]
macro_rules! custom_for {
    (let $var:ident = $val:expr; $cmp_var:ident < $y:expr; $inc_var:ident += $z:expr => {$body:expr}) => {
        {
            let init_val: i32 = $val;
            let y_val: i32 = $y;
            let z_val: usize = $z;
            for n in (init_val..y_val).step_by(z_val) {
                $body
            }
        }
    };
    (let $var:ident = $val:expr; $cmp_var:ident > $y:expr; $inc_var:ident -= $z:expr) => {
        {
            let init_val: i32 = $val;
            let y_val: i32 = $y;
            let z_val: usize = $z;
            for n in (y_val..init_val+1).step_by(z_val).rev() {
                println!("{}", n);    
            }
        }
    }
}

#[macro_export]
macro_rules! custom_reduce {
    // [$($elem:expr), *] means that it expects an array with 1 or more elements
    // custom_redude([1, 2, 3], item => acc + item, 0)
    // custom_redude([1, 2, 3], item => acc * item, 0)
    ([$($elem:expr), *], $item:ident => $acc:ident $operation:tt $item_usage:ident, $init_val:expr) => {{
        let mut $acc = $init_val;
        $(
            let $item = $elem;
            $acc = $acc $operation $elem; 
        )*
        $acc
    }};
}


