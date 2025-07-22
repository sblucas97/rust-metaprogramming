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


// TODO: implement custom reduce
/*
macro_rules! custom_reduce {

    ($acc:ident, $cur:ident => $acc_usage:ident + $cur_usage:ident, $init_val:ident) => {
        let mut acc = $init_val;
        

    }
}
*/

