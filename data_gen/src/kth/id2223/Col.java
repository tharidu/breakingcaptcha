package kth.id2223;

import nl.captcha.text.renderer.WordRenderer;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Shape;
import java.awt.font.FontRenderContext;
import java.awt.font.GlyphVector;
import java.awt.font.TextAttribute;
import java.awt.font.TextLayout;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.security.SecureRandom;
import java.text.AttributedCharacterIterator;
import java.text.AttributedString;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Col implements WordRenderer {

    private static final Random RAND = new SecureRandom();

    private static final List<Color> DEFAULT_COLORS = new ArrayList<Color>();
    private static final List<Font> DEFAULT_FONTS = new ArrayList<Font>();
    private static final float DEFAULT_STROKE_WIDTH = 0f;
    // The text will be rendered 25%/5% of the image height/width from the X and Y axes
    private static final double YOFFSET = 0.25;
    private static final double XOFFSET = 0.05;

    private final List<Font> _fonts;
    private final List<Color> _colors;
    private final float _strokeWidth;

    static {
        DEFAULT_FONTS.add(new Font("Arial", Font.BOLD, 40));
        DEFAULT_COLORS.add(Color.BLACK);
    }

    public Col() {
        this(DEFAULT_COLORS, DEFAULT_FONTS, DEFAULT_STROKE_WIDTH);
    }

    public Col(List<Color> colors, List<Font> fonts) {
        this(colors, fonts, DEFAULT_STROKE_WIDTH);
    }

    public Col(List<Color> colors, List<Font> fonts, float strokeWidth) {
        _colors = colors != null ? colors : DEFAULT_COLORS;
        _fonts = fonts != null ? fonts : DEFAULT_FONTS;
        _strokeWidth = strokeWidth < 0 ? DEFAULT_STROKE_WIDTH : strokeWidth;
    }

    @Override
    public void render(final String word, BufferedImage image) {
        Graphics2D g = image.createGraphics();

        RenderingHints hints = new RenderingHints(
                RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON);
        hints.add(new RenderingHints(RenderingHints.KEY_RENDERING,
                RenderingHints.VALUE_RENDER_QUALITY));
        g.setRenderingHints(hints);

        AttributedString as = new AttributedString(word);
        as.addAttribute(TextAttribute.FONT, getRandomFont());

        FontRenderContext frc;
        AttributedCharacterIterator aci;

        int xBaseline = (int) Math.round(image.getWidth() * XOFFSET);
        int yBaseline = image.getHeight() - (int) Math.round(image.getHeight() * YOFFSET);
        for (char c : word.toCharArray()) {
            as = new AttributedString(Character.toString(c));
            as.addAttribute(TextAttribute.FONT, getRandomFont());

            frc = g.getFontRenderContext();
            aci = as.getIterator();
            TextLayout tl1 = new TextLayout(aci, frc);

            Shape shape1;
            if (new Random().nextBoolean()) {
                shape1 = tl1.getOutline(AffineTransform.getTranslateInstance(xBaseline + RAND.nextInt(2), yBaseline + RAND.nextInt(10)));
                g.setColor(getRandomColor());
                g.setStroke(new BasicStroke(_strokeWidth));
                g.draw(shape1);
            } else {
                tl1.draw(g, xBaseline + RAND.nextInt(2), yBaseline + RAND.nextInt(10));
            }

            xBaseline = xBaseline + 20 - 2;
            yBaseline -= +RAND.nextInt(10);
        }

        g.setColor(getRandomColor());
        g.setStroke(new BasicStroke(_strokeWidth));

//        g.draw(shape);

//        shape = tl.getOutline(AffineTransform.getTranslateInstance(xBaseline + 5, yBaseline + 1));
//
//        g.setColor(getRandomColor());
//        g.setStroke(new BasicStroke(_strokeWidth));
//
//        g.draw(shape);
    }

    private Color getRandomColor() {
        return (Color) getRandomObject(_colors);
    }

    private Font getRandomFont() {
        return (Font) getRandomObject(_fonts);
    }

    private Object getRandomObject(List<? extends Object> objs) {
        if (objs.size() == 1) {
            return objs.get(0);
        }

        int i = RAND.nextInt(objs.size());
        return objs.get(i);
    }
}
