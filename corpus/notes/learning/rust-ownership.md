# Rust: ownership and borrowing

- Every value has one owner. When the owner goes out of scope the value is dropped.
- Assigning or passing a `String` moves it; the old variable can no longer be used.
  Types that are `Copy` (integers, bool, char) are copied instead.
- References borrow without taking ownership. Any number of `&T` at a time, or exactly
  one `&mut T`, never both. This is what prevents data races at compile time.
- Lifetimes describe how long a reference is valid. Most are inferred; you write them when
  a function returns a reference derived from one of several inputs.
- `Rc<T>` for shared ownership in one thread, `Arc<T>` across threads, `RefCell<T>` to
  move borrow checks to run time.

The error "borrowed value does not live long enough" usually means returning a reference
to something created inside the function. Return the owned value instead.
