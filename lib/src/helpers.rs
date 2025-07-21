pub fn get_headers() -> String {
    let stdio_header = r#"
    #include <stdio.h>
    "#;
    
    return stdio_header.to_string();
}

pub fn init_main_function() -> String {
    let main_function = r#"
        int main() {
    "#;

    return main_function.to_string();
}

pub fn close_main_function() -> String {
    let main_function = r#"
        return 0;
        }
    "#;

    return main_function.to_string();
}
