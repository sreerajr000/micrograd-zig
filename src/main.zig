const std = @import("std");

fn Context(comptime T: type) type {
    return struct {
        other: ?T = null,
        self: ?T = null,
        power: f32 = 1.0,
    };
}

fn Value(comptime T: type) type {
    return struct {
        const Self = @This();
        data: T,
        grad: T,
        backward: ?*const fn (*Self) void,
        prev: std.AutoHashMap(*Self, void), // Overkill right now, but keeping this for comparison
        op: []const u8 = "",
        allocator: std.mem.Allocator,
        context: Context(*Self),

        pub fn new(
            allocator: std.mem.Allocator,
            data: T,
            children: []const *Self,
            op: []const u8,
        ) !Self {
            var _prev = std.AutoHashMap(*Self, void).init(allocator);
            for (children) |child| {
                try _prev.put(child, {});
            }
            return Self{
                .data = data,
                .grad = 0.0,
                .backward = null,
                .prev = _prev,
                .op = op,
                .allocator = allocator,
                .context = Context(*Self){},
            };
        }

        pub fn add(self: *Self, other: *Self) !Self {
            var out = try Self.new(
                self.allocator,
                self.data + other.data,
                &[_]*Self{ self, other },
                "+",
            );
            out.context.other = other;
            out.context.self = self;
            out.backward = add_bwd;
            return out;
        }

        fn add_bwd(out: *Self) void {
            out.context.self.?.grad += out.grad;
            out.context.other.?.grad += out.grad;
        }

        pub fn mul(self: *Self, other: *Self) !Self {
            var out = try Self.new(
                self.allocator,
                self.data * other.data,
                &[_]*Self{ self, other },
                "*",
            );
            out.context.other = other;
            out.context.self = self;
            out.backward = mul_bwd;
            return out;
        }

        fn mul_bwd(out: *Self) void {
            out.context.self.?.grad += out.context.other.?.data * out.grad;
            out.context.other.?.grad == out.context.self.?.data * out.grad;
        }

        pub fn pow(self: *Self, other: T) !Self {
            var out = try Self.new(
                self.allocator,
                std.math.pow(T, self.data, other),
                &[_]*Self{self},
                "**",
            );
            out.context.power = other;
            out.context.self = self;
            out.backward = pow_bwd;
            return out;
        }

        fn pow_bwd(out: *Self) void {
            const other = out.context.power;
            var self = out.context.self.?;
            self.grad += (other * std.math.pow(T, self.data, (other - 1))) * out.grad;
        }

        pub fn relu(self: *Self) !Self {
            var out = try Self.new(
                self.allocator,
                if (self.data < 0) 0 else self.data,
                &[_]*Self{self},
                "ReLU",
            );
            out.context.self = self;
            out.backward = relu_bwd;
            return out;
        }

        fn relu_bwd(out: *Self) void {
            out.context.self.?.grad += (if (out.data > 0) 1 else 0) * out.grad;
        }

        pub fn print(self: *Self) void {
            std.debug.print("Content: [data: {any}, grad: {any}, op: {s}] \n", .{
                self.data,
                self.grad,
                self.op,
            });
        }

        pub fn deinit(self: *Self) void {
            self.prev.deinit();
        }
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var v = try Value(f32).new(allocator, 0.0, null, "*");
    v.print();
    v.deinit();
}

test "value" {
    // var allocator = std.testing.allocator;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var u = try Value(f32).new(allocator, 2.0, &[_]*Value(f32){}, "");
    var v = try Value(f32).new(allocator, 3.0, &[_]*Value(f32){}, "");
    try std.testing.expectEqual(@as(f32, 3.0), v.data);
    try std.testing.expectEqual(@as(f32, 2.0), u.data);

    var sum = try u.add(&v);
    try std.testing.expectEqual(@as(f32, 5.0), sum.data);
    // sum.backward = null;
    sum.backward.?(&sum);

    var pow = try u.pow(2.0);
    pow.grad = 1.0;
    pow.backward.?(&pow);
    std.debug.print(
        "u: {any}, v: {any}, u+v: {any}, u.grad: {}, v.grad: {}\n ",
        .{ u.data, v.data, sum.data, u.grad, v.grad },
    );

    std.debug.print("pow: {any}, grad: {any}\n", .{ pow.data, u.grad });
    sum.deinit();
    pow.deinit();
    try std.testing.expect(gpa.deinit() == std.heap.Check.ok);
}
